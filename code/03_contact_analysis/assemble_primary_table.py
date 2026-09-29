"""Pair direct-label threshold rows for the full-range extension.

New release adapter; does not estimate missing distances or change calls.
"""
import argparse
import csv
from pathlib import Path
from recompute_mask_proximity import PROTOCOL, require_completed_run

EXPECTED = {1: 2676, 2: 530, 3: 624}
THRESHOLDS = (30, 50, 80, 100)


def assemble(rows, cell, expected):
    grouped = {}
    for row in rows:
        if int(row['cell']) != cell:
            continue
        if row.get('measurement_protocol') != PROTOCOL:
            raise ValueError('Incompatible threshold table; use the current direct-label run.')
        ident = int(row['instance_id'])
        partner = row['partner']
        if partner not in ('er', 'golgi'):
            raise ValueError('Unexpected partner')
        group = grouped.setdefault(ident, {})
        if partner in group:
            raise ValueError('Duplicate instance/partner row')
        group[partner] = row
    if len(grouped) != expected:
        raise ValueError(f'Expected {expected} instances; got {len(grouped)}')
    output = []
    for ident, group in sorted(grouped.items()):
        if set(group) != {'er', 'golgi'}:
            raise ValueError(f'Incomplete partner pair for {ident}')
        if group['er']['morphology'] != group['golgi']['morphology']:
            raise ValueError(f'Morphology mismatch for {ident}')
        result = dict(cell=cell, instance_id=ident, morphology=group['er']['morphology'],
                      measurement_protocol=PROTOCOL)
        for partner, row in group.items():
            status = row['distance_status']
            value = row['min_distance_nm_le100']
            if status == 'resolved_le100_nm':
                distance = float(value)
                if not 0 <= distance <= 100 + 1e-9:
                    raise ValueError('Resolved distance outside threshold range')
            elif status == 'greater_than_100_nm':
                if str(value).strip():
                    raise ValueError('Censored distance must remain empty')
                distance = float('inf')
            else:
                raise ValueError('Unknown distance status')
            result[f'{partner}_distance_status'] = status
            result[f'{partner}_distance_nm_le100'] = value
            for threshold in THRESHOLDS:
                call = int(row[f'le_{threshold}nm'])
                if call != int(distance <= threshold + 1e-9):
                    raise ValueError('Distance/threshold inconsistency')
                result[f'{partner}_le_{threshold}nm'] = call
        result['golgi_missing_outer_strip_in_halo'] = group['golgi'].get('golgi_missing_outer_strip_in_halo', '')
        output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True, type=Path)
    parser.add_argument('--cell', required=True, type=int, choices=(1, 2, 3))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require_completed_run(args.previous / f'cell{args.cell}')
    with (args.previous / f'cell{args.cell}' / 'per_instance_proximity.csv').open(encoding='utf-8-sig', newline='') as f:
        records = assemble(list(csv.DictReader(f)), args.cell, EXPECTED[args.cell])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f'Wrote {len(records)} validated primary rows to {args.output}')


if __name__ == '__main__':
    main()
