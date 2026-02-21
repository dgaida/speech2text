import os
import re
import sys

VERSION_PARTS_COUNT = 3


def bump_version(version_str):
    """Increments the patch version of a version string (e.g., 0.1.0 -> 0.1.1)."""
    parts = version_str.split(".")
    if len(parts) != VERSION_PARTS_COUNT:
        # Handle cases like "0.1" by padding with zeros
        while len(parts) < VERSION_PARTS_COUNT:
            parts.append("0")
    major, minor, patch = map(int, parts)
    patch += 1
    return f"{major}.{minor}.{patch}"


def update_pyproject(filepath):
    """Updates the version in pyproject.toml and returns old and new versions."""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        sys.exit(1)

    with open(filepath) as f:
        content = f.read()

    # Match 'version = "X.Y.Z"' or 'version = 'X.Y.Z''
    pattern = r'(version\s*=\s*["\'])(\d+\.\d+\.\d+)(["\'])'
    match = re.search(pattern, content)
    if not match:
        # Fallback for 2-part version if needed
        pattern = r'(version\s*=\s*["\'])(\d+\.\d+)(["\'])'
        match = re.search(pattern, content)
        if not match:
            print("Error: Version string not found in pyproject.toml.")
            sys.exit(1)

    old_version = match.group(2)
    new_version = bump_version(old_version)

    # Replace the version string (only the first occurrence to avoid modifying dependencies)
    new_content = re.sub(pattern, rf"\g<1>{new_version}\g<3>", content, count=1)

    with open(filepath, "w") as f:
        f.write(new_content)

    return old_version, new_version


def update_readme(filepath, old_version, new_version):
    """Updates the version badge in README.md."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return

    with open(filepath) as f:
        content = f.read()

    # Match the shields.io badge URL part: version-X.Y.Z-blue
    # We look for the pattern: https://img.shields.io/badge/version-X.Y.Z-blue.svg
    badge_pattern = r"https://img\.shields\.io/badge/version-[\d\.]+-blue\.svg"
    new_badge = f"https://img.shields.io/badge/version-{new_version}-blue.svg"

    if re.search(badge_pattern, content):
        new_content = re.sub(badge_pattern, new_badge, content)
        with open(filepath, "w") as f:
            f.write(new_content)
        print(f"Updated README.md badge: {old_version} -> {new_version}")
    else:
        print("Warning: Version badge not found in README.md.")


if __name__ == "__main__":
    pyproject_path = "pyproject.toml"
    readme_path = "README.md"

    old_v, new_v = update_pyproject(pyproject_path)
    print(f"Updated {pyproject_path}: {old_v} -> {new_v}")

    update_readme(readme_path, old_v, new_v)

    # Output the new version for GitHub Actions to capture if needed
    print(new_v)
