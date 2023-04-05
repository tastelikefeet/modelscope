# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility
from modelscope.hub.errors import NotExistError
from modelscope.utils.logger import get_logger
import concurrent.futures

_future = None
logger = get_logger()


def _api_push_to_hub(repo_name, output_dir, token=None, private=True, commit_message=''):
    api = HubApi()
    api.login(token)
    try:
        api.get_model(repo_name)
    except Exception as e:
        if isinstance(e, NotExistError) or 'Not Found for url' in str(e):
            model_repo_url = api.create_model(
                model_id=repo_name,
                visibility=ModelVisibility.PUBLIC if not private else ModelVisibility.PRIVATE,
                license=Licenses.APACHE_V2,
                chinese_name=repo_name,
            )
            logger.info(f'Successfully create a model repo: {model_repo_url}')
        else:
            raise e

    api.push_model(
        model_id=repo_name,
        model_dir=output_dir,
        visibility=ModelVisibility.PUBLIC if not private else ModelVisibility.PRIVATE,
        chinese_name=repo_name,
        commit_message=commit_message,
    )
    commit_message = commit_message or 'No commit message'
    logger.info(f'Successfully upload the model to {repo_name} with message: {commit_message}')


def push_to_hub(repo_name, output_dir, token=None, private=True, retry=3, commit_message='', async_upload=False):
    """
    Args:
        repo_name: The repo name for the modelhub repo
        output_dir: The local output_dir for the checkpoint
        token: The user api token, function will check the `MODELSCOPE_API_TOKEN` variable if this argument is None
        private: If is a private repo, default True
        retry: Retry times if something error in uploading, default 3
        commit_message: The commit message
        async_upload: Upload async, if True, the `retry` parameter will have no affection
    """
    if token is None:
        token = os.environ.get('MODELSCOPE_API_TOKEN')
    assert token is not None, 'Either pass in a token or to set `MODELSCOPE_API_TOKEN` in the environment variables.'
    assert os.path.isdir(output_dir)
    assert 'configuration.json' in os.listdir(output_dir) or 'configuration.yaml' in os.listdir(output_dir) \
           or 'configuration.yml' in os.listdir(output_dir)

    if not async_upload:
        for i in range(retry):
            try:
                _api_push_to_hub(repo_name, output_dir, token, private, commit_message)
            except Exception as e:
                logger.info(f'Error happens when uploading model: {e}')
                continue
            break
    else:
        global _future
        if _future is not None and not _future.done():
            logger.error(f'Another uploading is running, '
                         f'this uploading to {repo_name} with message {commit_message} will be canceled.')
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            _future = executor.submit(_api_push_to_hub, repo_name, output_dir, token, private, commit_message)




