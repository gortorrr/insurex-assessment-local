import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class AnalysisArtifactContractTests(unittest.TestCase):
    def test_notebook_is_valid_utf8_thai_and_has_required_sections(self):
        notebook = json.loads((ROOT / "analysis/notebooks/insurex_analysis.ipynb").read_text(encoding="utf-8"))
        markdown = "".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        self.assertGreaterEqual(len([c for c in notebook["cells"] if c.get("cell_type") == "markdown"]), 4)
        self.assertIn("Explore and describe key information from the datasets", markdown)
        self.assertIn("ข้อจำกัด", markdown)
        self.assertNotIn("????", markdown)
        source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
        self.assertIn("raw_field_count = len(raw_columns)", source)
        self.assertIn("derived_fields = [\"income_band\"]", source)
        self.assertIn("fig_missing", source)
        self.assertIn("pair_quality", source)
        self.assertIn('df[left].notna() & df[right].notna() & (df[left] != df[right])', source)
        self.assertIn('"one_sided_missing"', source)

        errors = [
            output
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        self.assertEqual(errors, [])

    def test_aggregates_and_powerbi_use_same_acceptance_contract(self):
        aggregate = json.loads((ROOT / "analysis/data/derived/analysis_aggregates.json").read_text(encoding="utf-8"))
        contract = json.loads((ROOT / "analysis/shared/metric_contract.json").read_text(encoding="utf-8"))
        model = json.loads((ROOT / "analysis/powerbi/InsureX_Analysis.SemanticModel/model.bim").read_text(encoding="utf-8"))
        measures = {item["name"]: item["expression"] for item in model["model"]["tables"][0]["measures"]}

        self.assertEqual(aggregate["overall"]["records"], 215_993)
        self.assertEqual(aggregate["overall"]["accepted_records"], 2_622)
        self.assertEqual(contract["acceptance"]["denominator"], "rows with label in {0,1,2}")
        self.assertIn("REMOVEFILTERS('Campaign'[label])", measures["Valid Label Records"])
        self.assertIn("REMOVEFILTERS('Campaign'[label])", measures["Accepted Records"])
        self.assertIn("[Accepted Records]", measures["Offer Acceptance Rate"])
        self.assertIn("[Valid Label Records]", measures["Offer Acceptance Rate"])

    def test_power_query_types_all_original_fields_and_missing_rows_semantics(self):
        model = json.loads((ROOT / "analysis/powerbi/InsureX_Analysis.SemanticModel/model.bim").read_text(encoding="utf-8"))
        campaign = model["model"]["tables"][0]
        query = "\n".join(campaign["partitions"][0]["source"]["expression"])
        measures = {item["name"]: item["expression"] for item in campaign["measures"]}
        for field in [
            "campaign_month", "marital_sta", "main_occupation", "customer_segment", "gender",
            "have_acc_planet", "have_cc", "scb_payroll", "num_children", "age", "income",
            "maxosdc_last_30d", "dcspend_last_30d", "easypymt_last_30d", "savacc_bal",
            "currentacc_bal", "avg_savaccbal_30d", "avg_currentaccbal_30d", "mob", "inflow30d",
            "outflow30d", "inflow1_15", "outflow1_15", "net_flow_30d", "net_flow_15d", "label",
        ]:
            self.assertIn(f'"{field}"', query)
        self.assertIn('"Has Missing Original Field"', query)
        self.assertIn('Table.ReplaceValue(Promoted, "", null', query)
        self.assertIn("'Campaign'[Has Missing Original Field] = TRUE()", measures["Missing Rows"])
        self.assertNotIn("'Campaign'[label] = BLANK()", measures["Missing Rows"])

    def test_pair_quality_keeps_non_null_and_shared_missing_separate(self):
        quality = json.loads((ROOT / "analysis/data/derived/analysis_aggregates.json").read_text(encoding="utf-8"))["quality_checks"]
        self.assertEqual(quality["pair_equal_counts"]["inflow30d=inflow1_15"], 213_592)
        self.assertEqual(quality["pair_shared_missing_counts"]["inflow30d=inflow1_15"], 2_401)
        self.assertEqual(quality["pair_equal_including_shared_missing_counts"]["inflow30d=inflow1_15"], 215_993)


if __name__ == "__main__":
    unittest.main()
