package main

import (
	"github.com/xuri/excelize/v2"
	"path/filepath"
	"testing"
)

func TestTypedFormulaResults(t *testing.T) {
	f := excelize.NewFile()
	defer f.Close()
	tests := []struct {
		formula string
		want    any
	}{
		{"SUM(1,2,3)", float64(6)}, {"AVERAGE(10,20)", float64(15)}, {"COUNT(1,2,\"x\")", float64(2)}, {"COUNTA(1,\"x\")", float64(2)},
		{"MIN(2,3)", float64(2)}, {"MAX(2,3)", float64(3)}, {"IF(1>0,\"123\",\"no\")", "123"}, {"IF(1>0,\"\",\"no\")", ""},
		{"AND(TRUE,FALSE)", false}, {"OR(TRUE,FALSE)", true}, {`AND("FALSE",TRUE)`, false}, {`IF(OR(TRUE,FALSE),"ok","no")`, "ok"}, {"NOT(TRUE)", false}, {"ROUND(1.235,2)", 1.24}, {"ROUNDUP(1.231,2)", 1.24}, {"ROUNDDOWN(1.239,2)", 1.23}, {"ABS(-2)", float64(2)},
		{"IFERROR(1/0,\"missing\")", "missing"}, {"COUNTIF(B1:B2,\">1\")", float64(1)}, {"SUMIF(B1:B2,\">1\")", float64(2)},
	}
	f.SetCellValue("Sheet1", "B1", 1)
	f.SetCellValue("Sheet1", "B2", 2)
	for i, tc := range tests {
		cell, _ := excelize.CoordinatesToCellName(1, i+1)
		if err := f.SetCellFormula("Sheet1", cell, tc.formula); err != nil {
			t.Fatal(err)
		}
	}
	file := filepath.Join(t.TempDir(), "formulas.xlsx")
	if err := f.SaveAs(file); err != nil {
		t.Fatal(err)
	}
	values, err := calculate(file)
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != len(tests) {
		t.Fatalf("missing results: %d", len(values))
	}
	for i, value := range values {
		if value.Value != tests[i].want {
			t.Errorf("%s: got %#v, want %#v", tests[i].formula, value.Value, tests[i].want)
		}
	}
}
