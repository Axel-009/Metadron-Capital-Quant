"""Utility functions for working with Claude Skills and Files API.

This module provides helper functions for:
- Extracting file IDs from Claude API responses
- Downloading files via the Files API
- Saving files to disk
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from anthropic import Anthropic


def extract_file_ids(response) -> list[str]:
    """Extract all file IDs from a Claude API response."""
    file_ids = []

    for block in response.content:
        if block.type == "bash_code_execution_tool_result":
            try:
                if hasattr(block, "content") and hasattr(block.content, "content"):
                    for item in block.content.content:
                        if hasattr(item, "file_id"):
                            file_ids.append(item.file_id)
            except Exception:
                continue
        elif block.type == "tool_result":
            try:
                if hasattr(block, "output"):
                    output_str = str(block.output)
                    if "file_id" in output_str.lower():
                        try:
                            output_json = json.loads(output_str)
                            if isinstance(output_json, dict) and "file_id" in output_json:
                                file_ids.append(output_json["file_id"])
                            elif isinstance(output_json, list):
                                for item in output_json:
                                    if isinstance(item, dict) and "file_id" in item:
                                        file_ids.append(item["file_id"])
                        except json.JSONDecodeError:
                            pattern = r"file_id['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9_-]+)"
                            matches = re.findall(pattern, output_str)
                            file_ids.extend(matches)
            except Exception:
                continue

    seen = set()
    unique_file_ids = []
    for fid in file_ids:
        if fid not in seen:
            seen.add(fid)
            unique_file_ids.append(fid)
    return unique_file_ids


def download_file(
    client: Anthropic, file_id: str, output_path: str, overwrite: bool = True
) -> dict[str, Any]:
    """Download a file from Claude's Files API and save it locally."""
    result = {
        "file_id": file_id,
        "output_path": output_path,
        "size": 0,
        "success": False,
        "error": None,
    }

    try:
        file_exists = os.path.exists(output_path)
        if file_exists and not overwrite:
            result["error"] = f"File already exists: {output_path} (set overwrite=True to replace)"
            return result

        output_dir = os.path.dirname(output_path)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        file_content = client.beta.files.download(file_id=file_id)

        with open(output_path, "wb") as f:
            f.write(file_content.read())

        result["size"] = os.path.getsize(output_path)
        result["success"] = True
        result["overwritten"] = file_exists

    except Exception as e:
        result["error"] = str(e)

    return result


def download_all_files(
    client: Anthropic,
    response,
    output_dir: str = "outputs",
    prefix: str = "",
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    """Extract and download all files from a Claude API response."""
    file_ids = extract_file_ids(response)
    results = []

    for i, file_id in enumerate(file_ids, 1):
        try:
            file_info = client.beta.files.retrieve_metadata(file_id=file_id)
            filename = file_info.filename
        except Exception:
            filename = f"file_{i}.bin"

        if prefix:
            filename = f"{prefix}{filename}"

        output_path = os.path.join(output_dir, filename)
        result = download_file(client, file_id, output_path, overwrite=overwrite)
        results.append(result)

    return results


def get_file_info(client: Anthropic, file_id: str) -> dict[str, Any] | None:
    """Retrieve metadata about a file from the Files API."""
    try:
        file_info = client.beta.files.retrieve_metadata(file_id=file_id)
        return {
            "file_id": file_info.id,
            "filename": file_info.filename,
            "size": file_info.size_bytes,
            "mime_type": file_info.mime_type,
            "created_at": file_info.created_at,
            "type": file_info.type,
            "downloadable": file_info.downloadable,
        }
    except Exception as e:
        print(f"Error retrieving file info: {e}")
        return None
