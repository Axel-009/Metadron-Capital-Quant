"""Utility functions for managing custom skills with Claude's Skills API.

This module provides helper functions for:
- Creating and uploading custom skills
- Listing and retrieving skill information
- Managing skill versions
- Testing skills with Claude
- Deleting skills
"""

from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.lib import files_from_dir


def create_skill(client: Anthropic, skill_path: str, display_title: str) -> dict[str, Any]:
    """Create a new custom skill from a directory.

    The directory must contain:
    - SKILL.md file with YAML frontmatter (name, description)
    - Optional: scripts, resources, REFERENCE.md
    """
    try:
        skill_dir = Path(skill_path)
        if not skill_dir.exists():
            return {"success": False, "error": f"Skill directory does not exist: {skill_path}"}

        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return {"success": False, "error": f"SKILL.md not found in {skill_path}"}

        skill = client.beta.skills.create(
            display_title=display_title, files=files_from_dir(skill_path)
        )

        return {
            "success": True,
            "skill_id": skill.id,
            "display_title": skill.display_title,
            "latest_version": skill.latest_version,
            "created_at": skill.created_at,
            "source": skill.source,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def list_custom_skills(client: Anthropic) -> list[dict[str, Any]]:
    """List all custom skills in the workspace."""
    try:
        skills_response = client.beta.skills.list(source="custom")
        skills = []
        for skill in skills_response.data:
            skills.append(
                {
                    "skill_id": skill.id,
                    "display_title": skill.display_title,
                    "latest_version": skill.latest_version,
                    "created_at": skill.created_at,
                    "updated_at": skill.updated_at,
                }
            )
        return skills
    except Exception as e:
        print(f"Error listing skills: {e}")
        return []


def get_skill_version(
    client: Anthropic, skill_id: str, version: str = "latest"
) -> dict[str, Any] | None:
    """Get detailed information about a specific skill version."""
    try:
        if version == "latest":
            skill = client.beta.skills.retrieve(skill_id)
            version = skill.latest_version

        version_info = client.beta.skills.versions.retrieve(skill_id=skill_id, version=version)

        return {
            "version": version_info.version,
            "skill_id": version_info.skill_id,
            "name": version_info.name,
            "description": version_info.description,
            "directory": version_info.directory,
            "created_at": version_info.created_at,
        }
    except Exception as e:
        print(f"Error getting skill version: {e}")
        return None


def create_skill_version(client: Anthropic, skill_id: str, skill_path: str) -> dict[str, Any]:
    """Create a new version of an existing skill."""
    try:
        version = client.beta.skills.versions.create(
            skill_id=skill_id, files=files_from_dir(skill_path)
        )
        return {
            "success": True,
            "version": version.version,
            "skill_id": version.skill_id,
            "created_at": version.created_at,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_skill(client: Anthropic, skill_id: str, delete_versions: bool = True) -> bool:
    """Delete a custom skill and optionally all its versions."""
    try:
        if delete_versions:
            versions = client.beta.skills.versions.list(skill_id=skill_id)
            for version in versions.data:
                client.beta.skills.versions.delete(skill_id=skill_id, version=version.version)
        client.beta.skills.delete(skill_id)
        return True
    except Exception as e:
        print(f"Error deleting skill: {e}")
        return False


def test_skill(
    client: Anthropic,
    skill_id: str,
    test_prompt: str,
    model: str = "claude-sonnet-4-6",
    include_anthropic_skills: list[str] | None = None,
) -> Any:
    """Test a custom skill with a prompt."""
    skills = [{"type": "custom", "skill_id": skill_id, "version": "latest"}]
    if include_anthropic_skills:
        for anthropic_skill in include_anthropic_skills:
            skills.append({"type": "anthropic", "skill_id": anthropic_skill, "version": "latest"})

    response = client.beta.messages.create(
        model=model,
        max_tokens=4096,
        container={"skills": skills},
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        messages=[{"role": "user", "content": test_prompt}],
        betas=["code-execution-2025-08-25", "files-api-2025-04-14", "skills-2025-10-02"],
    )
    return response


def list_skill_versions(client: Anthropic, skill_id: str) -> list[dict[str, Any]]:
    """List all versions of a skill."""
    try:
        versions_response = client.beta.skills.versions.list(skill_id=skill_id)
        return [
            {
                "version": v.version,
                "skill_id": v.skill_id,
                "created_at": v.created_at,
            }
            for v in versions_response.data
        ]
    except Exception as e:
        print(f"Error listing versions: {e}")
        return []


def validate_skill_directory(skill_path: str) -> dict[str, Any]:
    """Validate a skill directory structure before upload."""
    result = {"valid": True, "errors": [], "warnings": [], "info": {}}
    skill_dir = Path(skill_path)

    if not skill_dir.exists():
        result["valid"] = False
        result["errors"].append(f"Directory does not exist: {skill_path}")
        return result

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        result["valid"] = False
        result["errors"].append("SKILL.md file is required")
    else:
        content = skill_md.read_text()
        if not content.startswith("---"):
            result["valid"] = False
            result["errors"].append("SKILL.md must start with YAML frontmatter (---)")
        else:
            try:
                end_idx = content.index("---", 3)
                frontmatter = content[3:end_idx].strip()
                if "name:" not in frontmatter:
                    result["valid"] = False
                    result["errors"].append("YAML frontmatter must include 'name' field")
                if "description:" not in frontmatter:
                    result["valid"] = False
                    result["errors"].append("YAML frontmatter must include 'description' field")
                if len(frontmatter) > 1024:
                    result["valid"] = False
                    result["errors"].append(
                        f"YAML frontmatter exceeds 1024 chars (found: {len(frontmatter)})"
                    )
            except ValueError:
                result["valid"] = False
                result["errors"].append("Invalid YAML frontmatter format")

    total_size = sum(f.stat().st_size for f in skill_dir.rglob("*") if f.is_file())
    result["info"]["total_size_mb"] = total_size / (1024 * 1024)
    if total_size > 8 * 1024 * 1024:
        result["valid"] = False
        result["errors"].append(
            f"Total size exceeds 8MB (found: {total_size / (1024 * 1024):.2f} MB)"
        )

    files = list(skill_dir.rglob("*"))
    result["info"]["file_count"] = len([f for f in files if f.is_file()])
    result["info"]["directory_count"] = len([f for f in files if f.is_dir()])

    if (skill_dir / "REFERENCE.md").exists():
        result["info"]["has_reference"] = True
    if (skill_dir / "scripts").exists():
        result["info"]["has_scripts"] = True
        result["info"]["script_files"] = [
            f.name for f in (skill_dir / "scripts").iterdir() if f.is_file()
        ]

    return result
