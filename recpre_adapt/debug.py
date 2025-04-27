from collections import defaultdict
import torch


def enable_memory_profiling():
    torch.cuda.memory._record_memory_history(enabled='all')


def top_allocation_sites(snapshot, top_n=10):
    site_sizes = defaultdict(int)

    for seg in snapshot.get('segments', []):
        for blk in seg.get('blocks', []):
            # only look at live allocations
            if blk.get('state') != 'active_allocated':
                continue

            # grab frames list (might be None or empty)
            frames = blk.get('frames') or []
            if not frames:
                # no trace info – skip or attribute to "<unknown>"
                continue

            # pick the first Python frame, or else the very first frame
            py_frame = next(
                (f for f in frames if f.get('filename', '').endswith('.py')),
                frames[0]
            )

            fn   = py_frame.get('filename', '<unknown>')
            ln   = py_frame.get('line',     0)
            nm   = py_frame.get('name',     '<unknown>')
            site = f"{fn}:{ln} ({nm})"

            # some versions use 'requested_size', others just 'size'
            size = blk.get('requested_size', blk.get('size', 0))
            site_sizes[site] += size

    # sort and print
    sorted_sites = sorted(site_sizes.items(), key=lambda x: x[1], reverse=True)
    print(f"Top {top_n} allocation sites:")
    for site, total_bytes in sorted_sites[:top_n]:
        mb = total_bytes / (1024**2)
        print(f"  {site:60s} → {mb:6.2f} MB")
