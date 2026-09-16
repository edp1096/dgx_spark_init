package main

import (
	"encoding/json"
	"fmt"
	"github.com/xuri/excelize/v2"
	"os"
	"strings"
)

type decoration struct {
	Sheets []struct {
		Name   string
		Charts []struct {
			Kind, Title, Cell, Categories string
			Series                        []struct{ Name, Values string }
		}
		Pivots []struct {
			Name, Source, Destination string
			Rows, Columns             []string
			Values                    []struct{ Field, Function string }
		}
		Shapes []struct{ Cell, Text, Kind string }
	}
}

func decorate(filename, spec string) error {
	data, err := os.ReadFile(spec)
	if err != nil {
		return err
	}
	var input decoration
	if err = json.Unmarshal(data, &input); err != nil {
		return err
	}
	f, err := excelize.OpenFile(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	kinds := map[string]excelize.ChartType{"bar": excelize.Col, "line": excelize.Line, "pie": excelize.Pie, "doughnut": excelize.Doughnut, "area": excelize.Area, "scatter": excelize.Scatter}
	for _, s := range input.Sheets {
		quote := "'" + strings.ReplaceAll(s.Name, "'", "''") + "'!"
		for _, c := range s.Charts {
			kind, ok := kinds[c.Kind]
			if !ok {
				return fmt.Errorf("unsupported chart kind")
			}
			series := []excelize.ChartSeries{}
			for _, v := range c.Series {
				series = append(series, excelize.ChartSeries{Name: v.Name, Categories: quote + c.Categories, Values: quote + v.Values})
			}
			if err = f.AddChart(s.Name, c.Cell, &excelize.Chart{Type: kind, Series: series, Title: excelize.ChartTitle{Paragraph: []excelize.RichTextRun{{Text: c.Title}}}, Legend: excelize.ChartLegend{Position: "bottom"}, Dimension: excelize.ChartDimension{Width: 640, Height: 360}}); err != nil {
				return err
			}
		}
		for _, p := range s.Pivots {
			rows := []excelize.PivotTableField{}
			cols := []excelize.PivotTableField{}
			values := []excelize.PivotTableField{}
			for _, n := range p.Rows {
				rows = append(rows, excelize.PivotTableField{Data: n})
			}
			for _, n := range p.Columns {
				cols = append(cols, excelize.PivotTableField{Data: n})
			}
			for _, v := range p.Values {
				values = append(values, excelize.PivotTableField{Data: v.Field, Subtotal: v.Function})
			}
			if err = f.AddPivotTable(&excelize.PivotTableOptions{DataRange: s.Name + "!" + p.Source, PivotTableRange: s.Name + "!" + p.Destination, Name: p.Name, Rows: rows, Columns: cols, Data: values, RowGrandTotals: true, ColGrandTotals: true, ShowRowHeaders: true, ShowColHeaders: true, PivotTableStyleName: "PivotStyleMedium9"}); err != nil {
				return err
			}
		}
		for _, s2 := range s.Shapes {
			kind := s2.Kind
			if kind == "arrow" {
				kind = "rightArrow"
			}
			if err = f.AddShape(s.Name, &excelize.Shape{Cell: s2.Cell, Type: kind, Width: 200, Height: 80, Paragraph: []excelize.RichTextRun{{Text: s2.Text}}, Fill: excelize.Fill{Color: []string{"E7EEF5"}}}); err != nil {
				return err
			}
		}
	}
	return f.Save()
}

func formattedCells(filename string) ([]result, error) {
	f, err := excelize.OpenFile(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var values []result
	for _, sheet := range f.GetSheetList() {
		rows, e := f.GetRows(sheet)
		if e != nil {
			return nil, e
		}
		for r, row := range rows {
			for c := range row {
				cell, _ := excelize.CoordinatesToCellName(c+1, r+1)
				value, e := f.GetCellValue(sheet, cell)
				if e != nil {
					return nil, e
				}
				values = append(values, result{Sheet: sheet, Cell: cell, Value: value})
			}
		}
	}
	return values, nil
}
