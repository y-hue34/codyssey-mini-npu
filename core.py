"""외부 라이브러리 없이 행렬, MAC, 판정과 반복 측정을 구현한다."""
import math
from time import perf_counter_ns

EPSILON = 1e-9


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('행렬 원소는 bool이 아닌 숫자여야 합니다.')
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError('표현 가능한 유한 숫자가 필요합니다.') from error
    if not math.isfinite(number):
        raise ValueError('NaN과 무한대는 사용할 수 없습니다.')
    return number


class Matrix:
    def __init__(self, rows, size=None):
        if not isinstance(rows, list) or not rows:
            raise ValueError('비어 있지 않은 2차원 배열이 필요합니다.')
        self.size = len(rows) if size is None else size
        if len(rows) != self.size:
            raise ValueError('행 개수 불일치: 기대 {}, 실제 {}'.format(self.size, len(rows)))
        self.rows = []
        for index, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != self.size:
                raise ValueError('{}행의 열 개수 불일치: {}개 필요'.format(index + 1, self.size))
            self.rows.append([finite_number(value) for value in row])

    def _check_position(self, row, column):
        if type(row) is not int or type(column) is not int or not (0 <= row < self.size and 0 <= column < self.size):
            raise IndexError('행/열 위치는 0~{} 정수여야 합니다.'.format(self.size - 1))

    def get(self, row, column):
        self._check_position(row, column)
        return self.rows[row][column]

    def set(self, row, column, value):
        self._check_position(row, column)
        self.rows[row][column] = finite_number(value)


def normalize_label(value):
    if not isinstance(value, str):
        raise ValueError('라벨은 문자열이어야 합니다.')
    labels = {'+': 'Cross', 'cross': 'Cross', 'x': 'X'}
    key = value.strip().lower()
    if key not in labels:
        raise ValueError('알 수 없는 라벨: {!r} (Cross/+/X/x만 허용)'.format(value))
    return labels[key]


def mac(pattern, kernel):
    if pattern.size != kernel.size:
        raise ValueError('패턴과 필터 크기가 다릅니다.')
    total = 0.0
    for row in range(pattern.size):
        for column in range(pattern.size):
            total += pattern.rows[row][column] * kernel.rows[row][column]
    if not math.isfinite(total):
        raise ArithmeticError('MAC 결과가 표현 범위를 벗어났습니다.')
    return total


def compare_scores(score_a, score_b, label_a='Cross', label_b='X', epsilon=EPSILON):
    score_a, score_b = finite_number(score_a), finite_number(score_b)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon은 유한한 양수여야 합니다.')
    if abs(score_a - score_b) < epsilon:
        return 'UNDECIDED'
    return label_a if score_a > score_b else label_b


def benchmark(pattern, kernel, repeats=30):
    """1회 워밍업 후 MAC 호출만 repeats회 측정하여 평균 ms 반환."""
    if type(repeats) is not int or repeats < 10:
        raise ValueError('최소 10회 반복 측정해야 합니다.')
    mac(pattern, kernel)
    elapsed = 0
    for _ in range(repeats):
        start = perf_counter_ns()
        mac(pattern, kernel)
        elapsed += perf_counter_ns() - start
    return elapsed / repeats / 1_000_000


def generate_pattern(size, label):
    """홀수 크기의 십자가 또는 X 패턴을 생성한다 (선택 과제)."""
    if type(size) is not int or size <= 0 or size % 2 == 0:
        raise ValueError('크기는 양의 홀수여야 합니다.')
    label = normalize_label(label)
    middle = size // 2
    rows = []
    for row in range(size):
        values = []
        for column in range(size):
            active = (row == middle or column == middle) if label == 'Cross' else (row == column or row + column == size - 1)
            values.append(1.0 if active else 0.0)
        rows.append(values)
    return Matrix(rows)
