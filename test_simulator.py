import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from analysis import analyze, load_document, summarize
from core import Matrix, mac, compare_scores, benchmark, generate_pattern, normalize_label, EPSILON
from main import main, read_matrix


class SimulatorTests(unittest.TestCase):
    def fixture(self):
        return load_document(Path(__file__).resolve().parent / 'data.json')

    def test_known_cross_and_x_scores(self):
        cross, x = generate_pattern(3, 'Cross'), generate_pattern(3, 'X')
        self.assertEqual(mac(cross, cross), 5)
        self.assertEqual(mac(cross, x), 1)
        self.assertEqual(compare_scores(1, 5, 'A', 'B'), 'B')

    def test_all_required_sizes_and_position_access(self):
        for size in (3, 5, 13, 25):
            matrix = generate_pattern(size, 'Cross')
            self.assertEqual(mac(matrix, matrix), 2 * size - 1)
            matrix.set(0, 0, 0.25)
            self.assertEqual(matrix.get(0, 0), 0.25)
            with self.assertRaises(IndexError):
                matrix.get(-1, 0)

    def test_label_aliases(self):
        for label in ('+', 'cross', 'Cross', ' CROSS '):
            self.assertEqual(normalize_label(label), 'Cross')
        for label in ('X', 'x', ' x '):
            self.assertEqual(normalize_label(label), 'X')
        for value in ('unknown', 'UNDECIDED', None, 0):
            with self.assertRaises(ValueError):
                normalize_label(value)

    def test_epsilon_strict_boundary(self):
        self.assertEqual(compare_scores(0.9, 0.8999999999999999), 'UNDECIDED')
        self.assertEqual(compare_scores(0, EPSILON), 'X')  # < epsilon, not <=
        self.assertEqual(compare_scores(0, EPSILON / 2), 'UNDECIDED')
        self.assertEqual(compare_scores(-3, -2), 'X')

    def test_matrix_rejects_invalid_values_and_dimensions(self):
        for rows in ([], [[1, 2]], [[1], [2]], [[True]], [[float('nan')]], [[float('inf')]], [['1']]):
            with self.assertRaises(ValueError):
                Matrix(rows)
        with self.assertRaises(ValueError):
            mac(Matrix([[1]]), generate_pattern(3, 'X'))

    def test_mac_overflow_is_reported(self):
        with self.assertRaises(ArithmeticError):
            mac(Matrix([[1e308]]), Matrix([[1e308]]))

    def test_provided_data_results(self):
        results = analyze(self.fixture())
        self.assertEqual(summarize(results), (6, 3, 3))
        self.assertEqual([r['id'] for r in results if not r['passed']], ['size_5_1', 'size_13_2', 'size_25_1'])
        self.assertTrue(all(r['prediction'] == 'UNDECIDED' for r in results if not r['passed']))

    def test_bad_case_does_not_stop_remaining_cases(self):
        data = self.fixture()
        data['patterns']['size_5_1']['input'] = [[1]]
        results = analyze(data)
        self.assertEqual(len(results), 6)
        self.assertIn('행 개수 불일치', results[0]['reason'])
        self.assertTrue(results[1]['passed'])

    def test_unknown_expected_and_bad_key_are_case_failures(self):
        data = self.fixture()
        data['patterns']['size_5_2']['expected'] = 'triangle'
        data['patterns']['not_size_key'] = data['patterns']['size_25_2']
        results = analyze(data)
        self.assertEqual(len(results), 7)
        self.assertFalse(results[1]['passed'])
        self.assertIn('알 수 없는 라벨', results[1]['reason'])
        self.assertIn('키 형식 오류', results[-1]['reason'])

    def test_missing_and_duplicate_filters(self):
        data = self.fixture()
        del data['filters']['size_5']
        results = analyze(data)
        self.assertFalse(results[0]['passed'])
        self.assertFalse(results[1]['passed'])
        self.assertTrue(results[2]['passed'])
        data = self.fixture()
        data['filters']['size_5']['+'] = data['filters']['size_5']['cross']
        self.assertIn('중복', analyze(data)[0]['reason'])

    def test_filters_accept_standard_case(self):
        data = self.fixture()
        for filters in data['filters'].values():
            filters['Cross'] = filters.pop('cross')
            filters['X'] = filters.pop('x')
        self.assertEqual(summarize(analyze(data)), (6, 3, 3))

    def test_benchmark_repetition_count_and_unit(self):
        pattern = generate_pattern(3, 'Cross')
        with patch('core.mac', wraps=mac) as operation, patch('core.perf_counter_ns', side_effect=[0, 1000] * 10):
            self.assertAlmostEqual(benchmark(pattern, pattern, 10), 0.001)
            self.assertEqual(operation.call_count, 11)  # warmup + 10
        with self.assertRaises(ValueError):
            benchmark(pattern, pattern, 9)

    def test_manual_input_retries_only_bad_row(self):
        with patch('builtins.input', side_effect=['', '1 0', 'a b c', 'nan 0 1', '1 0 1', '0 1 0', '1 0 1']), contextlib.redirect_stdout(io.StringIO()) as output:
            matrix = read_matrix('패턴')
        self.assertEqual(mac(matrix, matrix), 5)
        self.assertEqual(output.getvalue().count('입력 형식 오류'), 4)

    def test_interrupt_and_eof_exit_cleanly(self):
        for error in (EOFError, KeyboardInterrupt):
            with patch('builtins.input', side_effect=error), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main([]), 0)
            self.assertIn('안전하게 종료', output.getvalue())

    def test_missing_file_has_clear_error(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['--mode', 'json', '--data', 'no-such-data.json']), 1)
        self.assertIn('분석 오류', output.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
