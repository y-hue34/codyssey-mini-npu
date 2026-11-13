"""Mini NPU Simulator: Python 3.8+, 표준 라이브러리만 사용."""
import argparse
from pathlib import Path

from analysis import load_document, analyze, summarize, validate_filters
from core import Matrix, mac, compare_scores, benchmark, generate_pattern


def read_matrix(name, size=3):
    rows = []
    print('\n{}: {}줄, 각 줄에 {}개 숫자'.format(name, size, size))
    while len(rows) < size:
        text = input('{}행: '.format(len(rows) + 1)).strip()
        try:
            tokens = text.split()
            if len(tokens) != size:
                raise ValueError('열 개수 불일치')
            row = [float(value) for value in tokens]
            # Matrix의 공통 검증으로 NaN/Infinity도 거절한다.
            Matrix([row[:] for _ in range(size)], size)
            rows.append(row)
        except (ValueError, OverflowError):
            print('입력 형식 오류: 각 줄에 {}개의 유한한 숫자를 공백으로 구분해 입력하세요. 이 행을 다시 입력하세요.'.format(size))
    return Matrix(rows, size)


def print_performance(document=None, pattern=None, kernel=None, repeats=30):
    print('\n[성능 분석] 워밍업 1회 제외, 각 크기 MAC {}회 평균'.format(repeats))
    print('크기(N×N) | 평균 시간(ms) | 연산 횟수(N², MAC 1회 기준) | 데이터')
    sizes = (3,) if document is None else (3, 5, 13, 25)
    for size in sizes:
        try:
            if document is None:
                sample, selected = pattern, kernel
                source = '입력 패턴/필터 A'
            elif size == 3:
                sample = selected = generate_pattern(3, 'Cross')
                source = '생성 Cross (제공 파일에 3×3 없음)'
            else:
                selected = validate_filters(document['filters'].get('size_{}'.format(size)), size)['Cross']
                sample = generate_pattern(size, 'Cross')
                source = '생성 Cross / 제공 Cross 필터'
            average = benchmark(sample, selected, repeats)
            print('{}×{} | {:.9f} | {} | {}'.format(size, size, average, size * size, source))
        except (ValueError, TypeError, ArithmeticError) as error:
            print('{}×{} | 측정 불가 | {} | {}'.format(size, size, size * size, error))


def manual_mode(repeats):
    first = read_matrix('필터 A')
    second = read_matrix('필터 B')
    print('필터 A/B 저장 확인 (메모리 3×3 행렬)')
    for name, matrix in (('A', first), ('B', second)):
        print(name + ':')
        for row in matrix.rows:
            print(' '.join('{:g}'.format(value) for value in row))
    pattern = read_matrix('패턴')
    score_a, score_b = mac(pattern, first), mac(pattern, second)
    outcome = compare_scores(score_a, score_b, 'A', 'B')
    print('\n[MAC 결과]')
    print('A 점수: {:.17g}\nB 점수: {:.17g}'.format(score_a, score_b))
    average_a = benchmark(pattern, first, repeats)
    average_b = benchmark(pattern, second, repeats)
    print('연산 시간 (각 필터 {}회 평균): A {:.9f} ms, B {:.9f} ms'.format(repeats, average_a, average_b))
    print('판정: {}'.format('판정 불가 (UNDECIDED, |A-B| < 1e-9)' if outcome == 'UNDECIDED' else outcome))
    print_performance(pattern=pattern, kernel=first, repeats=repeats)


def json_mode(path, repeats):
    print('\n[data.json 분석] ' + str(path))
    document = load_document(path)
    print('[필터 로드]')
    for size in (5, 13, 25):
        try:
            validate_filters(document['filters'].get('size_{}'.format(size)), size)
            print('size_{}: Cross, X 필터 로드 완료'.format(size))
        except (ValueError, TypeError) as error:
            print('size_{}: 필터 검증 실패 - {}'.format(size, error))
    print('[패턴 로드/검증 및 MAC 판정]')
    results = analyze(document)
    for result in results:
        print('\n--- {} ---'.format(result['id']))
        if result['scores'] is not None:
            print('Cross 점수: {:.17g} | X 점수: {:.17g}'.format(result['scores']['Cross'], result['scores']['X']))
        print('판정: {} | expected: {} | {}'.format(result['prediction'] or '계산 불가', result['expected'] or '검증 불가', 'PASS' if result['passed'] else 'FAIL'))
        if result['reason']:
            print('사유 [{}]: {}'.format(result['category'], result['reason']))
    print_performance(document=document, repeats=repeats)
    total, passed, failed = summarize(results)
    print('\n[결과 요약]\n총 테스트: {}개\n통과: {}개\n실패: {}개'.format(total, passed, failed))
    if failed:
        print('실패 케이스:')
        for result in results:
            if not result['passed']:
                print('- {}: {}'.format(result['id'], result['reason']))
    else:
        print('실패 케이스: 없음')
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description='MAC 기반 Mini NPU Simulator')
    parser.add_argument('--mode', choices=['manual', 'json'], help='생략하면 콘솔에서 선택')
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent / 'data.json')
    parser.add_argument('--repeats', type=int, default=30, help='MAC 반복 측정 횟수 (최소 10)')
    args = parser.parse_args(argv)
    if args.repeats < 10:
        parser.error('--repeats는 10 이상이어야 합니다.')
    print('=== Mini NPU Simulator ===')
    try:
        mode = args.mode
        while mode is None:
            print('1. 사용자 입력 (3×3)\n2. data.json 분석\n0. 종료')
            choice = input('선택: ').strip()
            if choice == '0':
                return 0
            mode = {'1': 'manual', '2': 'json'}.get(choice)
            if mode is None:
                print('0, 1, 2 중 하나를 입력하세요.')
        if mode == 'manual':
            manual_mode(args.repeats)
        else:
            json_mode(args.data, args.repeats)
        return 0  # 케이스 FAIL과 프로그램 실행 실패는 구분한다.
    except (KeyboardInterrupt, EOFError):
        print('\n입력이 중단되어 안전하게 종료합니다. 미완성 행렬은 판정하지 않습니다.')
        return 0
    except (OSError, UnicodeError, ValueError, ArithmeticError, RecursionError) as error:
        print('분석 오류: {}'.format(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
