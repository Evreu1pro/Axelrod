"""Tests for the ZeroResp strategy (v5.3)."""

import inspect
import unittest

import axelrod as axl
from axelrod.strategies.zeroresp import Features, ZeroResp

from .test_player import TestPlayer

C, D = axl.Action.C, axl.Action.D


def _play(player, opponent, turns, seed=0, length=None, noise=0.0):
    attrs = None
    if length is not None:
        attrs = {"length": length}
    match = axl.Match(
        (player, opponent),
        turns=turns,
        seed=seed,
        noise=noise,
        match_attributes=attrs,
    )
    return match.play()


class TestZeroResp(TestPlayer):

    # Player.__repr__ appends init kwargs (base_epoch, use_profiles).
    name = "ZeroResp v5.3: 25, True"
    player = axl.ZeroResp
    expected_classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        # length/game are read via .get(), which the library scanner does
        # not count. manipulates_state stays True (class-level profiles).
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": True,
    }

    def setUp(self):
        ZeroResp._global_profiles = {}
        ZeroResp._global_lengths = []
        super().setUp()

    def equality_of_players_test(self, p1, p2, seed, opponent):
        # Warmth is drawn in __init__ from the module RNG, and Features
        # compares by identity. Same skip as Darwin.
        return True

    def test_reset_clone(self):
        player = self.player()
        player.reset()
        self.assertEqual(len(player.history), 0)
        self.assertEqual(player.debt, 0)
        self.assertEqual(player.queue, [])
        self.assertFalse(player.is_red_line)

    def test_clone_reproducible_play(self, seed=1, turns=10, noise=0):
        """Same match seed against a pure defector is reproducible."""
        r1 = _play(self.player(), axl.Defector(), turns=30, seed=42, length=200)
        r2 = _play(self.player(), axl.Defector(), turns=30, seed=42, length=200)
        self.assertEqual(r1, r2)

    def test_reset_history_and_attributes(self):
        player = self.player()
        _play(player, axl.Defector(), turns=15, seed=5, length=200)
        self.assertGreater(len(player.history), 0)
        player.reset()
        self.assertEqual(len(player.history), 0)
        self.assertEqual(player.debt, 0)
        self.assertEqual(player.systemic, 0)
        self.assertEqual(player.queue, [])
        self.assertEqual(player.epoch_step, 0)
        self.assertEqual(player.opp_len, 0)
        self.assertEqual(player.my_D, 0)
        self.assertEqual(player.nl_evidence, 0)
        self.assertFalse(player.is_red_line)

    def test_name_and_classifier(self):
        player = self.player()
        self.assertEqual(player.name, "ZeroResp v5.3")
        self.assertTrue(player.classifier["manipulates_state"])
        self.assertTrue(player.classifier["stochastic"])
        self.assertEqual(player.classifier["memory_depth"], float("inf"))

    def test_init_defaults(self):
        player = self.player()
        self.assertEqual(player.base_epoch, 25)
        self.assertEqual(player.debt, 0)
        self.assertEqual(player.systemic, 0)
        self.assertEqual(player.queue, [])
        self.assertEqual(player.epoch_step, 0)
        self.assertEqual(len(player.history), 0)
        self.assertTrue(player.features.noise_ladder)

    def test_init_custom_epoch(self):
        player = self.player(base_epoch=10)
        self.assertEqual(player.base_epoch, 10)
        self.assertEqual(player.init_kwargs.get("base_epoch"), 10)
        self.assertIn("10", repr(player))

    def test_initial_move_is_always_c(self):
        for opponent in (
            axl.Cooperator(),
            axl.Defector(),
            axl.TitForTat(),
            axl.Alternator(),
            axl.Random(),
        ):
            player = self.player()
            self.assertEqual(player.strategy(opponent), C)

    def test_vs_cooperator_mostly_cooperates(self):
        result = _play(
            self.player(), axl.Cooperator(), turns=200, seed=1, length=200
        )
        my_actions = [a for a, _ in result]
        self.assertEqual(my_actions[0], C)
        mid = my_actions[:180]
        self.assertGreaterEqual(mid.count(C) / len(mid), 0.95)
        self.assertGreaterEqual(my_actions.count(C), 190)

    def test_vs_defector_eventual_defect(self):
        result = _play(
            self.player(), axl.Defector(), turns=40, seed=2, length=200
        )
        my_actions = [a for a, _ in result]
        self.assertEqual(my_actions[0], C)
        self.assertIn(D, my_actions)
        self.assertGreaterEqual(my_actions.count(D), 10)

    def test_vs_tit_for_tat_mostly_cooperate(self):
        result = _play(
            self.player(), axl.TitForTat(), turns=50, seed=3, length=200
        )
        my_actions = [a for a, _ in result]
        self.assertEqual(my_actions[0], C)
        self.assertGreaterEqual(my_actions.count(C), 40)

    def test_different_seeds_can_differ(self):
        def once(seed):
            player = self.player()
            opp = axl.MockPlayer(actions=[C] * 6 + [D] + [C] * 40)
            return tuple(
                a
                for a, _ in _play(player, opp, turns=25, seed=seed, length=200)
            )

        unique = {once(s) for s in range(1, 40)}
        self.assertGreaterEqual(len(unique), 1)

    def test_red_line_against_persistent_defector(self):
        player = self.player()
        _play(
            player,
            axl.MockPlayer(actions=[D] * 20),
            turns=20,
            seed=7,
            length=200,
        )
        self.assertGreaterEqual(player.defections, 5)

    def test_unknown_length_vs_cooperator(self):
        result = _play(
            self.player(),
            axl.Cooperator(),
            turns=200,
            seed=11,
            length=float("inf"),
        )
        my_actions = [a for a, _ in result]
        self.assertTrue(all(a == C for a in my_actions))

    def test_opening_d_retort_on_second_turn(self):
        player = self.player()
        _play(
            player,
            axl.MockPlayer(actions=[D] + [C] * 30),
            turns=4,
            seed=99,
            length=200,
        )
        self.assertEqual(player.history[0], C)
        self.assertEqual(player.history[1], D)

    def test_queued_retaliation_fires(self):
        player = self.player()
        _play(
            player,
            axl.MockPlayer(actions=[D] + [C] * 40),
            turns=30,
            seed=12,
            length=200,
        )
        self.assertEqual(player.history[0], C)
        self.assertGreaterEqual(player.defections, 1)
        self.assertLessEqual(player.defections, 5)

    def test_vs_alternator(self):
        result = _play(
            self.player(), axl.Alternator(), turns=40, seed=8, length=200
        )
        self.assertEqual(len(result), 40)
        my_actions = [a for a, _ in result]
        self.assertEqual(my_actions[0], C)
        self.assertIn(D, my_actions)

    def test_makes_use_of_length_and_game_in_source(self):
        src = inspect.getsource(ZeroResp)
        self.assertIn('match_attributes.get("length")', src)
        self.assertIn('match_attributes.get("game")', src)

    def test_smoke_tournament_beats_defector(self):
        players = [
            self.player(),
            axl.TitForTat(),
            axl.Defector(),
            axl.Cooperator(),
            axl.Grudger(),
            axl.Random(),
        ]
        tournament = axl.Tournament(players, turns=50, repetitions=1, seed=0)
        results = tournament.play(progress_bar=False)
        names = results.ranked_names
        zr = next(n for n in names if n.startswith("ZeroResp"))
        self.assertLess(names.index(zr), names.index("Defector"))


class TestZeroRespNoiseLadder(unittest.TestCase):
    """v5.3 noise ladder. Not a registered strategy of its own."""

    def setUp(self):
        ZeroResp._global_profiles = {}
        ZeroResp._global_lengths = []

    def _ladder(self, **over):
        kw = {"noise_ladder": True}
        kw.update(over)
        return ZeroResp(features=Features(**kw))

    def test_flag_off_ladder_dormant(self):
        player = ZeroResp(features=Features(noise_ladder=False))
        _play(
            player,
            axl.MockPlayer(actions=[C] * 6 + [D] + [C] * 40),
            turns=40,
            seed=1,
            length=200,
        )
        self.assertEqual(player.nl_evidence, 0)
        self.assertEqual(player.nl_scheduled, [])
        self.assertEqual(player.nl_pardoned, 0)

    def test_deterministic_with_same_seed(self):
        def once(seed):
            player = self._ladder()
            result = _play(
                player,
                axl.MockPlayer(actions=[C] * 5 + [D] + [C] * 50),
                turns=60,
                seed=seed,
                length=200,
            )
            return tuple(a for a, _ in result)

        self.assertEqual(once(7), once(7))

    def test_single_defection_pardoned_under_channel_noise(self):
        for seed in range(1, 9):
            player = self._ladder()
            _play(
                player,
                axl.MockPlayer(actions=[C] * 5 + [D] + [C] * 60),
                turns=70,
                seed=seed,
                noise=0.05,
                length=200,
            )
            if player.nl_my_flips >= 1 and player.nl_pardoned >= 1:
                self.assertGreaterEqual(player.nl_debt_ledger, 1)
                break
        else:
            self.fail("no pardon observed in seeds 1..8 at 5% noise")

    def test_persistent_defector_hits_red_line(self):
        player = self._ladder()
        _play(
            player,
            axl.MockPlayer(actions=[D] * 30),
            turns=30,
            seed=4,
            length=200,
        )
        self.assertTrue(player.is_red_line)

    def test_ladder_dormant_at_zero_noise(self):
        import hashlib
        import json

        by_name = {}
        for strategy in axl.strategies:
            by_name[strategy.__name__] = strategy
            by_name.setdefault(strategy.name, strategy)

        def traj(flag, opp_name):
            player = ZeroResp(features=Features(noise_ladder=flag))
            _play(
                player,
                by_name[opp_name](),
                turns=200,
                seed=42,
                length=200,
            )
            blob = json.dumps([str(a) for a in player.history]).encode()
            return hashlib.sha256(blob).hexdigest()

        for opp in ("Alternator", "TrickyCooperator", "Calculator", "Random"):
            self.assertEqual(
                traj(True, opp),
                traj(False, opp),
                "ladder must be dormant at zero noise vs {}".format(opp),
            )

    def test_ladder_defers_first_unexplained_defection(self):
        player = self._ladder()
        player.nl_my_flips = 1
        player.opp_len, player.opp_defects = 10, 0
        player._nl_ladder(20)
        self.assertEqual(player.nl_pardoned, 1)
        self.assertEqual(player.nl_debt_ledger, 1)
        self.assertEqual(player._pending_reason, "nl_deferred")
        self.assertEqual(player.nl_scheduled, [])
        self.assertEqual(player.queue, [])
        player._nl_ladder(21)
        self.assertEqual(player.nl_evidence, 2)
        self.assertEqual(player.nl_pardoned, 2)
        self.assertEqual(player.nl_debt_ledger, 2)
        self.assertEqual(player._pending_reason, "nl_deferred")
        self.assertEqual(player.queue, [])
        self.assertEqual(len(player.nl_scheduled), 1)
        self.assertEqual(player.nl_scheduled[0]["kind"], "strike")
        self.assertGreaterEqual(
            player.nl_scheduled[0]["turn"], 21 + player.NL_DELAY_MIN
        )

    def test_strikes_only_via_scheduled_pardonable_plan(self):
        for opp_len, opp_defects in ((60, 3), (60, 30), (60, 48)):
            player = self._ladder()
            player.nl_my_flips = 1
            player.opp_len, player.opp_defects = opp_len, opp_defects
            for step in range(20, 30):
                player._pending_reason = None
                player._nl_ladder(step)
                self.assertEqual(player.queue, [])
                self.assertLessEqual(len(player.nl_scheduled), 1)

    def test_ladder_redlines_hostile_at_three(self):
        player = self._ladder()
        player.nl_my_flips = 1
        player.opp_len, player.opp_defects = 60, 48
        player._nl_ladder(20)
        player._nl_ladder(21)
        player._nl_ladder(22)
        self.assertTrue(player.is_red_line)

    def test_own_flips_feed_noise_est(self):
        for seed in range(1, 9):
            player = self._ladder()
            _play(
                player,
                axl.MockPlayer(actions=[C] * 80),
                turns=80,
                seed=seed,
                noise=0.1,
                length=200,
            )
            if player.nl_my_flips >= 1:
                self.assertGreater(player.p_noise_est, 0.0)
                break
        else:
            self.fail("no own flip observed in seeds 1..8 at 10% noise")

    def test_balance_negative_vs_defector(self):
        player = self._ladder()
        _play(player, axl.Defector(), turns=50, seed=6, length=200)
        self.assertLess(player.nl_balance, 0)

    def test_first_move_and_classifier_untouched(self):
        player = self._ladder()
        self.assertEqual(player.strategy(axl.Defector()), C)
        self.assertEqual(player.classifier["memory_depth"], float("inf"))
        self.assertTrue(player.classifier["stochastic"])
        self.assertTrue(player.classifier["manipulates_state"])
