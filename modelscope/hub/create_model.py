import tempfile
from typing import Any, Dict, Optional

import json
from requests.exceptions import HTTPError

from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.hub.constants import ModelVisibility
from modelscope.utils.logger import get_logger
from .utils.utils import add_patterns_to_file, add_patterns_to_gitattributes

logger = get_logger()


def create_model_repo(repo_id: str,
                      token: Optional[str] = None,
                      private: bool = False,
                      config_json: Optional[Dict[str, Any]] = None) -> str:
    """Create model repo and create .gitattributes file and .gitignore file

    Args:
        repo_id(str): The repo id
        token(str, Optional): The access token of the user
        private(bool): If is a private repo, default False
        config_json(Dict[str, Any]): An optional config_json to fill into the configuration.json file,
            If None, the default content will be uploaded:
            ```json
                {"framework": "pytorch", "task": "text-generation", "allow_remote": True}
            ```
            You can manually modify this in the modelhub.
    """
    api = HubApi()
    assert repo_id is not None, 'Please enter a valid repo id'
    api.try_login(token)
    visibility = ModelVisibility.PRIVATE if private else ModelVisibility.PUBLIC
    if '/' not in repo_id:
        user_name = ModelScopeConfig.get_user_info()[0]
        assert isinstance(user_name, str)
        repo_id = f'{user_name}/{repo_id}'
        logger.info(
            f"'/' not in hub_model_id, pushing to personal repo {repo_id}")
    try:
        api.create_model(repo_id, visibility)
    except HTTPError:
        # The remote repository has been created
        pass

    with tempfile.TemporaryDirectory() as temp_cache_dir:
        from modelscope.hub.repository import Repository
        repo = Repository(temp_cache_dir, repo_id)
        add_patterns_to_gitattributes(
            repo, ['*.safetensors', '*.bin', '*.pt', '*.gguf'])
        default_config = {
            'framework': 'pytorch',
            'task': 'text-generation',
            'allow_remote': True
        }
        if not config_json:
            config_json = {}
        config = {**default_config, **config_json}
        add_patterns_to_file(
            repo,
            'configuration.json', [json.dumps(config)],
            ignore_push_error=True)
    return repo_id