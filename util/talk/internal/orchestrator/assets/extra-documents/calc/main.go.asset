package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/xuri/excelize/v2"
)

type result struct {
	Sheet string `json:"sheet"`
	Cell  string `json:"cell"`
	Value any    `json:"value"`
}

func calculate(filename string) ([]result, error) {
	f, err := excelize.OpenFile(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var cells []result
	for _, sheet := range f.GetSheetList() {
		rows, err := f.GetRows(sheet)
		if err != nil {
			return nil, err
		}
		for r, row := range rows {
			for c := range row {
				cell, _ := excelize.CoordinatesToCellName(c+1, r+1)
				formula, err := f.GetCellFormula(sheet, cell)
				if err != nil {
					return nil, err
				}
				if formula != "" {
					cells = append(cells, result{Sheet: sheet, Cell: cell})
				}
			}
		}
	}
	scratch := "__sparktalk_types"
	for {
		idx, _ := f.GetSheetIndex(scratch)
		if idx < 0 {
			break
		}
		scratch += "_"
		if len(scratch) > 31 {
			return nil, fmt.Errorf("cannot allocate calculation sheet")
		}
	}
	if _, err = f.NewSheet(scratch); err != nil {
		return nil, err
	}
	for i, c := range cells {
		ref := "'" + strings.ReplaceAll(c.Sheet, "'", "''") + "'!" + c.Cell
		if err = f.SetCellFormula(scratch, fmt.Sprintf("A%d", i+1), "TYPE("+ref+")"); err != nil {
			return nil, err
		}
	}
	for i, c := range cells {
		value, err := f.CalcCellValue(c.Sheet, c.Cell, excelize.Options{RawCellValue: true})
		if err != nil {
			return nil, fmt.Errorf("Formula calculation failed: %s!%s: %w", c.Sheet, c.Cell, err)
		}
		kind, err := f.CalcCellValue(scratch, fmt.Sprintf("A%d", i+1), excelize.Options{RawCellValue: true})
		if err != nil {
			return nil, err
		}
		switch kind {
		case "1":
			number, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, err
			}
			cells[i].Value = number
		case "2":
			cells[i].Value = value
		case "4":
			switch strings.ToUpper(value) {
			case "TRUE", "1":
				cells[i].Value = true
			case "FALSE", "0":
				cells[i].Value = false
			default:
				return nil, fmt.Errorf("unexpected boolean result")
			}
		default:
			return nil, fmt.Errorf("unsupported formula result type %s at %s!%s", kind, c.Sheet, c.Cell)
		}
	}
	return cells, nil
}
func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: document-calc generated.xlsx")
		os.Exit(2)
	}
	values, err := calculate(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if values == nil {
		values = []result{}
	}
	if err = json.NewEncoder(os.Stdout).Encode(values); err != nil {
		os.Exit(1)
	}
}
