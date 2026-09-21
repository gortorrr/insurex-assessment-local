import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POWERBI = ROOT / "analysis" / "powerbi"


class PowerBIProjectArtifactTests(unittest.TestCase):
    def test_pbip_points_to_report_and_local_model(self):
        project = json.loads((POWERBI / "InsureX_Analysis.pbip").read_text(encoding="utf-8"))
        report_folder = project["artifacts"][0]["report"]["path"]
        self.assertEqual(report_folder, "InsureX_Analysis.Report")

        pbir = json.loads((POWERBI / report_folder / "definition.pbir").read_text(encoding="utf-8"))
        self.assertEqual(pbir["version"], "4.0")
        self.assertEqual(pbir["datasetReference"]["byPath"]["path"], "../InsureX_Analysis.SemanticModel")
        self.assertNotIn("byConnection", pbir["datasetReference"])
        self.assertTrue((POWERBI / report_folder / ".platform").exists())
        self.assertFalse((POWERBI / report_folder / "item.config.json").exists())

    def test_model_has_raw_and_derived_fields_and_contract_measures(self):
        model = json.loads((POWERBI / "InsureX_Analysis.SemanticModel" / "model.bim").read_text(encoding="utf-8"))
        campaign = model["model"]["tables"][0]
        # Desktop requires 1606 or later for PBIP external-change reload.
        self.assertGreaterEqual(model["compatibilityLevel"], 1606)
        field_names = {column["name"] for column in campaign["columns"]}
        self.assertEqual(len(field_names & {
            "campaign_month", "marital_sta", "main_occupation", "customer_segment", "gender",
            "have_acc_planet", "have_cc", "scb_payroll", "num_children", "age", "income",
            "maxosdc_last_30d", "dcspend_last_30d", "easypymt_last_30d", "savacc_bal",
            "currentacc_bal", "avg_savaccbal_30d", "avg_currentaccbal_30d", "mob", "inflow30d",
            "outflow30d", "inflow1_15", "outflow1_15", "net_flow_30d", "net_flow_15d", "label",
        }), 26)
        self.assertIn("Has Missing Original Field", field_names)
        self.assertIn("income_band", field_names)
        self.assertIn("campaign_month_order", field_names)
        self.assertIn("response_label", field_names)
        campaign_month = next(column for column in campaign["columns"] if column["name"] == "campaign_month")
        self.assertEqual(campaign_month["sortByColumn"], "campaign_month_order")
        pbism = json.loads((POWERBI / "InsureX_Analysis.SemanticModel" / "definition.pbism").read_text(encoding="utf-8"))
        self.assertEqual(pbism["version"], "4.2")
        self.assertTrue((POWERBI / "InsureX_Analysis.SemanticModel" / ".platform").exists())
        self.assertFalse((POWERBI / "InsureX_Analysis.SemanticModel" / "item.config.json").exists())
        measure_names = {measure["name"] for measure in campaign["measures"]}
        self.assertTrue({"Valid Label Records", "Accepted Records", "Offer Acceptance Rate", "Missing Rows"} <= measure_names)

    def test_report_has_required_pages_and_visual_bindings(self):
        report = json.loads((POWERBI / "InsureX_Analysis.Report" / "report.json").read_text(encoding="utf-8"))
        # Desktop's themeCurrent handler reads themeCollection.customTheme.
        # Valid JSON alone does not protect against this rendering failure.
        self.assertIsInstance(json.loads(report["config"])["themeCollection"], dict)
        # Desktop omits an empty resourcePackages collection when saving.
        self.assertIsInstance(report.get("resourcePackages", []), list)
        self.assertEqual(json.loads(report["filters"]), [])
        pages = {section["displayName"] for section in report["sections"]}
        self.assertEqual(pages, {"Dashboard"})
        self.assertGreaterEqual(sum(len(section["visualContainers"]) for section in report["sections"]), 10)
        raw_config = json.dumps(report, ensure_ascii=False)
        self.assertIn("Offer Acceptance Rate", raw_config)
        self.assertIn("Missing Rows", raw_config)
        self.assertIn("Campaign.response_label", raw_config)
        visual_types = [
            json.loads(visual["config"]).get("singleVisual", {}).get("visualType")
            for section in report["sections"] for visual in section["visualContainers"]
        ]
        self.assertEqual(visual_types.count("slicer"), 3)
        self.assertEqual(visual_types.count("card"), 4)


if __name__ == "__main__":
    unittest.main()
