from datetime import date
import unittest

from insurex.kpi import (
    KpiRule,
    MonthlyPerformance,
    evaluate_series,
    synthetic_fixture,
    validate_rules,
)


RULE = KpiRule(rule_id="v1", valid_from_month=date(2026, 1, 1))


def performance(month: int, premium: int, policies: int) -> MonthlyPerformance:
    return MonthlyPerformance(
        agent_id="A",
        month_start=date(2026, month, 1),
        total_premium_satang=premium,
        new_policy_count=policies,
    )


class KpiTests(unittest.TestCase):
    def test_strict_boundaries(self):
        cases = [
            (1500000, 6, "FAIL"),
            (1500001, 5, "FAIL"),
            (1500001, 6, "PASS"),
        ]
        for premium, policies, expected in cases:
            evaluations, _ = evaluate_series(
                agent_id="A",
                initial_contract_type="salary",
                performances=[performance(1, premium, policies)],
                rules=[RULE],
            )
            self.assertEqual(evaluations[0].result, expected)

    def test_fail_streak_transitions_salary_to_commission_next_month(self):
        evaluations, events = evaluate_series(
            agent_id="A",
            initial_contract_type="salary",
            performances=[performance(1, 1500000, 6), performance(2, 1500000, 6), performance(3, 1500000, 6), performance(4, 1500000, 6)],
            rules=[RULE],
        )
        self.assertEqual([row.fail_streak for row in evaluations], [1, 2, 3, 4])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].to_type, "commission")
        self.assertEqual(events[0].effective_from, date(2026, 4, 1))

    def test_pass_streak_transitions_commission_to_salary(self):
        _, events = evaluate_series(
            agent_id="A",
            initial_contract_type="commission",
            performances=[performance(1, 1500001, 6), performance(2, 1500001, 6), performance(3, 1500001, 6), performance(4, 1500001, 6)],
            rules=[RULE],
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].to_type, "salary")
        self.assertEqual(events[0].effective_from, date(2026, 4, 1))

    def test_opposite_result_resets_streak(self):
        evaluations, events = evaluate_series(
            agent_id="A",
            initial_contract_type="salary",
            performances=[performance(1, 1500001, 6), performance(2, 1500001, 6), performance(3, 1500000, 6), performance(4, 1500001, 6)],
            rules=[RULE],
        )
        self.assertEqual([row.pass_streak for row in evaluations], [1, 2, 0, 1])
        self.assertEqual([row.fail_streak for row in evaluations], [0, 0, 1, 0])
        self.assertEqual(events, [])

    def test_missing_calendar_month_does_not_bridge_streak(self):
        evaluations, _ = evaluate_series(
            agent_id="A",
            initial_contract_type="salary",
            performances=[performance(1, 1500001, 6), performance(3, 1500001, 6), performance(4, 1500001, 6)],
            rules=[RULE],
        )
        self.assertEqual([row.pass_streak for row in evaluations], [1, 1, 2])

    def test_pending_does_not_bridge_streak(self):
        pending = MonthlyPerformance(
            agent_id="A",
            month_start=date(2026, 2, 1),
            total_premium_satang=None,
            new_policy_count=None,
            data_status="PENDING",
        )
        evaluations, _ = evaluate_series(
            agent_id="A",
            initial_contract_type="salary",
            performances=[performance(1, 1500001, 6), pending, performance(3, 1500001, 6)],
            rules=[RULE],
        )
        self.assertEqual([row.result for row in evaluations], ["PASS", "PENDING", "PASS"])
        self.assertEqual([row.pass_streak for row in evaluations], [1, 0, 1])

    def test_year_boundary_is_consecutive(self):
        rule = KpiRule(rule_id="v1", valid_from_month=date(2025, 1, 1))
        rows = [
            MonthlyPerformance("A", date(2025, 11, 1), 1500001, 6),
            MonthlyPerformance("A", date(2025, 12, 1), 1500001, 6),
            MonthlyPerformance("A", date(2026, 1, 1), 1500001, 6),
        ]
        evaluations, _ = evaluate_series(
            agent_id="A", initial_contract_type="commission", performances=rows, rules=[rule]
        )
        self.assertEqual(evaluations[-1].pass_streak, 3)

    def test_duplicate_month_and_rule_overlap_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_series(
                agent_id="A",
                initial_contract_type="salary",
                performances=[performance(1, 1500001, 6), performance(1, 1500001, 6)],
                rules=[RULE],
            )
        with self.assertRaises(ValueError):
            validate_rules(
                [
                    KpiRule("v1", date(2026, 1, 1), date(2026, 6, 1)),
                    KpiRule("v2", date(2026, 6, 1), None),
                ]
            )

    def test_synthetic_fixture_covers_both_transition_directions(self):
        agents, rules, rows = synthetic_fixture()
        all_events = []
        for agent in agents:
            agent_rows = [row for row in rows if row.agent_id == agent["agent_id"]]
            _, events = evaluate_series(
                agent_id=agent["agent_id"],
                initial_contract_type=agent["initial_contract_type"],
                performances=agent_rows,
                rules=rules,
            )
            all_events.extend(events)
        self.assertIn(("SYN-A", "commission"), [(e.agent_id, e.to_type) for e in all_events])
        self.assertIn(("SYN-B", "salary"), [(e.agent_id, e.to_type) for e in all_events])

    def test_joined_mid_month_and_open_month_are_pending(self):
        evaluations, _ = evaluate_series(
            agent_id="A",
            initial_contract_type="salary",
            joined_on=date(2026, 1, 15),
            closed_through=date(2026, 1, 31),
            performances=[performance(1, 1500001, 6), performance(2, 1500001, 6)],
            rules=[RULE],
        )
        self.assertEqual([row.result for row in evaluations], ["PENDING", "PENDING"])

    def test_performance_before_joined_or_after_inactive_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "before agent joined"):
            evaluate_series(
                agent_id="A", initial_contract_type="salary", joined_on=date(2026, 2, 1),
                performances=[performance(1, 1500001, 6)], rules=[RULE]
            )
        with self.assertRaisesRegex(ValueError, "after agent inactive"):
            evaluate_series(
                agent_id="A", initial_contract_type="salary", inactive_on=date(2026, 1, 15),
                performances=[performance(2, 1500001, 6)], rules=[RULE]
            )

    def test_agent_id_and_measure_types_are_validated(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            evaluate_series(
                agent_id="A", initial_contract_type="salary",
                performances=[MonthlyPerformance("B", date(2026, 1, 1), 1500001, 6)], rules=[RULE]
            )
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            evaluate_series(
                agent_id="A", initial_contract_type="salary",
                performances=[MonthlyPerformance("A", date(2026, 1, 1), 1500000.5, 6)], rules=[RULE]
            )


if __name__ == "__main__":
    unittest.main()
