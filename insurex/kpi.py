"""Deterministic monthly KPI evaluation for the synthetic demonstration data."""

from __future__ import annotations

import calendar
import hashlib
import json
from dataclasses import dataclass
from calendar import monthrange
from datetime import date
from typing import Iterable


VALID_STATUS = {"COMPLETE", "PENDING"}
VALID_CONTRACTS = {"salary", "commission"}


@dataclass(frozen=True)
class KpiRule:
    rule_id: str
    valid_from_month: date
    valid_to_month: date | None = None
    premium_threshold_satang: int = 1_500_000
    policy_threshold: int = 5
    required_streak: int = 3
    comparison_operator: str = ">"

    def covers(self, month_start: date) -> bool:
        return self.valid_from_month <= month_start and (
            self.valid_to_month is None or month_start <= self.valid_to_month
        )


@dataclass(frozen=True)
class MonthlyPerformance:
    agent_id: str
    month_start: date
    total_premium_satang: int | None
    new_policy_count: int | None
    data_status: str = "COMPLETE"
    source_kind: str = "synthetic"
    month_closed: bool = True


@dataclass(frozen=True)
class Evaluation:
    agent_id: str
    month_start: date
    rule_id: str | None
    result: str
    pass_streak: int
    fail_streak: int
    input_hash: str


@dataclass(frozen=True)
class ContractEvent:
    agent_id: str
    from_type: str
    to_type: str
    effective_from: date
    trigger_month: date
    reason: str


def month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def next_month(value: date) -> date:
    year = value.year + (value.month == 12)
    month = 1 if value.month == 12 else value.month + 1
    return date(year, month, 1)


def month_end(value: date) -> date:
    return date(value.year, value.month, monthrange(value.year, value.month)[1])


def month_distance(left: date, right: date) -> int:
    return (right.year - left.year) * 12 + right.month - left.month


def validate_rules(rules: Iterable[KpiRule]) -> list[KpiRule]:
    ordered = sorted(rules, key=lambda item: item.valid_from_month)
    if not ordered:
        raise ValueError("at least one KPI rule is required")
    for rule in ordered:
        if rule.comparison_operator != ">":
            raise ValueError("only strict > comparison is supported")
        if rule.required_streak <= 0 or rule.premium_threshold_satang < 0 or rule.policy_threshold < 0:
            raise ValueError("invalid KPI rule threshold or streak")
        if rule.valid_to_month and rule.valid_to_month < rule.valid_from_month:
            raise ValueError("rule valid_to_month precedes valid_from_month")
    for previous, current in zip(ordered, ordered[1:]):
        if previous.valid_to_month is None or previous.valid_to_month >= current.valid_from_month:
            raise ValueError(f"overlapping KPI rules: {previous.rule_id}, {current.rule_id}")
    return ordered


def select_rule(rules: Iterable[KpiRule], month: date) -> KpiRule:
    matches = [rule for rule in rules if rule.covers(month)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one KPI rule for {month.isoformat()}, found {len(matches)}")
    return matches[0]


def input_hash(performance: MonthlyPerformance) -> str:
    payload = {
        "agent_id": performance.agent_id,
        "month_start": performance.month_start.isoformat(),
        "total_premium_satang": performance.total_premium_satang,
        "new_policy_count": performance.new_policy_count,
        "data_status": performance.data_status,
        "source_kind": performance.source_kind,
        "month_closed": performance.month_closed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _validate_performances(
    performances: Iterable[MonthlyPerformance],
    *,
    expected_agent_id: str,
    joined_on: date | None,
    inactive_on: date | None,
) -> list[MonthlyPerformance]:
    rows = sorted(performances, key=lambda item: item.month_start)
    seen: set[date] = set()
    for row in rows:
        if row.agent_id != expected_agent_id:
            raise ValueError(f"performance agent_id {row.agent_id} does not match {expected_agent_id}")
        if row.month_start != month_start(row.month_start):
            raise ValueError("month_start must be the first day of a month")
        if row.month_start in seen:
            raise ValueError(f"duplicate performance month: {row.month_start.isoformat()}")
        seen.add(row.month_start)
        if row.data_status not in VALID_STATUS:
            raise ValueError(f"invalid data status: {row.data_status}")
        if row.source_kind not in {"synthetic", "external"}:
            raise ValueError(f"invalid source kind: {row.source_kind}")
        for field_name, value in {
            "total_premium_satang": row.total_premium_satang,
            "new_policy_count": row.new_policy_count,
        }.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"{field_name} must be an integer or None")
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if row.data_status == "COMPLETE":
            if row.total_premium_satang is None or row.new_policy_count is None:
                raise ValueError("COMPLETE performance requires premium and policy count")
        if joined_on is not None and row.month_start < month_start(joined_on):
            raise ValueError("performance before agent joined_on cannot be evaluated")
        if inactive_on is not None and row.month_start > month_start(inactive_on):
            raise ValueError("performance after agent inactive_on cannot be evaluated")
    return rows


def evaluate_series(
    *,
    agent_id: str,
    initial_contract_type: str,
    performances: Iterable[MonthlyPerformance],
    rules: Iterable[KpiRule],
    joined_on: date | None = None,
    inactive_on: date | None = None,
    closed_through: date | None = None,
) -> tuple[list[Evaluation], list[ContractEvent]]:
    if initial_contract_type not in VALID_CONTRACTS:
        raise ValueError("initial_contract_type must be salary or commission")
    validated_rules = validate_rules(rules)
    rows = _validate_performances(
        performances,
        expected_agent_id=agent_id,
        joined_on=joined_on,
        inactive_on=inactive_on,
    )
    evaluations: list[Evaluation] = []
    events: list[ContractEvent] = []
    pass_streak = 0
    fail_streak = 0
    previous_month: date | None = None
    previous_result: str | None = None
    previous_rule: str | None = None
    contract = initial_contract_type

    for performance in rows:
        rule = select_rule(validated_rules, performance.month_start)
        contiguous = previous_month is not None and month_distance(previous_month, performance.month_start) == 1
        same_rule = previous_rule == rule.rule_id
        can_continue = contiguous and same_rule and previous_result in {"PASS", "FAIL"}
        if not can_continue:
            pass_streak = 0
            fail_streak = 0

        month_is_open = not performance.month_closed
        if closed_through is not None and performance.month_start > month_start(closed_through):
            month_is_open = True
        if joined_on is not None and performance.month_start == month_start(joined_on) and joined_on.day > 1:
            month_is_open = True
        if inactive_on is not None and performance.month_start == month_start(inactive_on) and inactive_on < month_end(inactive_on):
            month_is_open = True

        if performance.data_status == "PENDING" or month_is_open:
            result = "PENDING"
            pass_streak = 0
            fail_streak = 0
        else:
            assert performance.total_premium_satang is not None
            assert performance.new_policy_count is not None
            passed = (
                performance.total_premium_satang > rule.premium_threshold_satang
                and performance.new_policy_count > rule.policy_threshold
            )
            result = "PASS" if passed else "FAIL"
            if result == "PASS":
                pass_streak = pass_streak + 1 if can_continue and previous_result == "PASS" else 1
                fail_streak = 0
            else:
                fail_streak = fail_streak + 1 if can_continue and previous_result == "FAIL" else 1
                pass_streak = 0

        evaluation = Evaluation(
            agent_id=agent_id,
            month_start=performance.month_start,
            rule_id=rule.rule_id,
            result=result,
            pass_streak=pass_streak,
            fail_streak=fail_streak,
            input_hash=input_hash(performance),
        )
        evaluations.append(evaluation)

        target: str | None = None
        if result == "FAIL" and fail_streak >= rule.required_streak and contract == "salary":
            target = "commission"
        if result == "PASS" and pass_streak >= rule.required_streak and contract == "commission":
            target = "salary"
        if target:
            events.append(
                ContractEvent(
                    agent_id=agent_id,
                    from_type=contract,
                    to_type=target,
                    effective_from=next_month(performance.month_start),
                    trigger_month=performance.month_start,
                    reason=f"{result.lower()} streak reached {rule.required_streak} consecutive complete months",
                )
            )
            contract = target

        previous_month = performance.month_start
        previous_result = result
        previous_rule = rule.rule_id

    return evaluations, events


def synthetic_fixture() -> tuple[list[dict[str, str]], list[KpiRule], list[MonthlyPerformance]]:
    """Return four deterministic agents and eight labelled synthetic months."""

    rule = KpiRule(rule_id="kpi-v1", valid_from_month=date(2026, 1, 1))
    agents = [
        {"agent_id": "SYN-A", "agent_name": "Synthetic Fail Agent", "initial_contract_type": "salary"},
        {"agent_id": "SYN-B", "agent_name": "Synthetic Pass Agent", "initial_contract_type": "commission"},
        {"agent_id": "SYN-C", "agent_name": "Synthetic Reset Agent", "initial_contract_type": "salary"},
        {"agent_id": "SYN-D", "agent_name": "Synthetic Gap Agent", "initial_contract_type": "salary"},
    ]
    patterns = {
        "SYN-A": [(1500000, 6), (1500000, 6), (1500000, 6), (1500000, 6), (1600000, 6), (1600000, 6), (1600000, 6), (1600000, 6)],
        "SYN-B": [(1600000, 6), (1600000, 6), (1600000, 6), (1600000, 6), (1500000, 5), (1500000, 5), (1500000, 5), (1500000, 5)],
        "SYN-C": [(1600000, 6), (1600000, 6), (1500000, 6), (1600000, 6), (1600000, 6), (1600000, 6), (1500000, 5), (1600000, 6)],
        "SYN-D": [(1600000, 6), None, (1600000, 6), (1600000, 6), (1600000, 6), (1500000, 5), (1500000, 5), (1500000, 5)],
    }
    performances: list[MonthlyPerformance] = []
    for agent_id, months in patterns.items():
        for index, values in enumerate(months, start=1):
            if values is None:
                continue
            performances.append(
                MonthlyPerformance(
                    agent_id=agent_id,
                    month_start=date(2026, index, 1),
                    total_premium_satang=values[0],
                    new_policy_count=values[1],
                    source_kind="synthetic",
                )
            )
    return agents, [rule], performances
