#!/usr/bin/env python3
"""
Custom git describe for setuptools_scm that handles tags with + (like v0.2.6+rocm.1)

Output format:
- When distance > 0 and tag contains '+': base+local.devN-0-ghash
  (embeds .devN in local part, sets distance to 0 to prevent version_scheme modification)
- Otherwise: tag-distance-ghash (standard git describe --long format)
"""

import subprocess
import sys


def main():
    try:
        # Get all tags sorted by version
        result = subprocess.run(
            ["git", "tag", "-l", "v*", "--sort=-version:refname"],
            capture_output=True,
            text=True,
            check=True,
        )

        if not result.stdout.strip():
            print("v0.0.0-0-g0000000")
            sys.exit(1)

        # Closest ancestor tag, tracked separately for tags that carry a local
        # version (`v0.5.3+amd.2`) and those that do not. Merging an upstream
        # release makes its bare tag an ancestor and usually the *closer* one,
        # and a bare tag produces a version indistinguishable from upstream's
        # own wheel -- no `+amd`, and no error to say so. Fork tags therefore
        # win outright; plain tags are the fallback for a tree that has none.
        best = {True: (None, float("inf")), False: (None, float("inf"))}

        for tag in result.stdout.strip().split("\n"):
            try:
                # Check if this tag is an ancestor of HEAD
                subprocess.run(
                    ["git", "merge-base", "--is-ancestor", tag, "HEAD"],
                    check=True,
                    capture_output=True,
                )

                # Get distance from this tag
                distance_result = subprocess.run(
                    ["git", "rev-list", f"{tag}..HEAD", "--count"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                distance = int(distance_result.stdout.strip())

                # Track the closest tag within its own bucket
                bucket = "+" in tag
                if distance <= best[bucket][1]:
                    best[bucket] = (tag, distance)

            except subprocess.CalledProcessError:
                # Not an ancestor, try next tag
                continue

        closest_tag, min_distance = best[True] if best[True][0] else best[False]

        if closest_tag is None:
            # No suitable tag found
            print("v0.0.0-0-g0000000")
            return 1

        # Get current commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit_hash = hash_result.stdout.strip()

        # If there are commits after the tag, embed .devN in the local part
        if min_distance > 0 and "+" in closest_tag:
            # Split tag into base and local: v0.2.6+rocm.1 -> v0.2.6, rocm.1
            base, local = closest_tag.split("+", 1)
            modified_tag = f"{base}+{local}.dev{min_distance}"
            # Output with distance set to 0 to prevent version_scheme from modifying it
            print(f"{modified_tag}-0-g{commit_hash}")
        else:
            # No local part or on exact tag - standard format
            print(f"{closest_tag}-{min_distance}-g{commit_hash}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
