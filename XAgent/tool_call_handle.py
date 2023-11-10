import abc
import os
import time
import base64
import uuid
import json5 as json
import requests

from typing import List
from colorama import Fore, Style
from concurrent.futures import ThreadPoolExecutor

from XAgent.config import CONFIG
from XAgent.logs import logger
from XAgent.data_structure.node import ToolNode
from XAgent.utils import ToolCallStatusCode
from XAgent.running_recorder import recorder
from XAgent.ai_functions import function_manager

def is_wrapped_response(obj: dict) -> bool:
    """
    Check if the response object is wrapped.

    Args:
        obj (dict): The response object.

    Returns:
        bool: True if the response object is wrapped, False otherwise.
    """
    return (
        'type' in obj
        and obj['type'] in ['simple', 'composite', 'binary']
        and 'data' in obj
    )

def unwrap_tool_response(obj):
    """
    Unwrap the tool response object.

    Args:
        obj: The tool response object.

    Returns:
        The unwrapped tool response object.
    """
    if isinstance(obj, dict):
        if not is_wrapped_response(obj):
            return obj
        match obj['type']:
            case 'simple':
                return obj['data']
            case 'binary':
                name = obj.get('name', uuid.uuid4().hex)
                if obj['media_type'] == 'image/png' and not str(name).endswith('.png'):
                    name += '.png'
                with open(os.path.join('local_workspace', name), 'wb') as f:
                    f.write(base64.b64decode(obj['data']))
                return {
                    'media_type': obj['media_type'],
                    'file_name': name
                }
            case 'composite':
                return [unwrap_tool_response(o) for o in obj['data']]
    elif isinstance(obj, (str, int, float, bool, list)):
        return obj
    elif obj is None:
        return None
    else:
        logger.typewriter_log(f'Unknown type {type(obj)} in unwrap_tool_response', Fore.YELLOW)
        return None


class ToolServerInterface():
    """
    The interface to communicate with the ToolServer.
    """

    def __init__(self):
        pass

    def lazy_init(self, config):
        """
        Lazy initialization of the ToolServer interface.

        Args:
            config: The configuration for the ToolServer.

        Raises:
            NotImplementedError: If trying to use a non-selfhost ToolServer.
        """
        self.config = config
        if config.use_selfhost_toolserver:
            self.url = config.selfhost_toolserver_url
        else:
            raise NotImplementedError('Please use selfhost toolserver')
        logger.typewriter_log("ToolServer connected in", Fore.RED, self.url)
        response = requests.post(f'{self.url}/get_cookie',)
        self.cookies = response.cookies

    def close(self):
        """
        Close the ToolServer session.
        """
        requests.post(f'{self.url}/close_session', cookies=self.cookies)

    def upload_file(self, file_path) -> str:
        """
        Upload a file to the ToolServer.

        Args:
            file_path (str): The path to the file to be uploaded.

        Returns:
            str: The response from the ToolServer.
        """
        url  = f"{self.url}/upload_file"
        response = requests.post(url, timeout=10, cookies=self.cookies,
                                 files={'file': open(file_path, 'rb'), 'filename': os.path.basename(file_path)})
        response.raise_for_status()
        response = response.json()
        return response

    def download_file(self, file_path) -> str:
        """
        Download a file from the ToolServer.

        Args:
            file_path (str): The path to the file to be downloaded.

        Returns:
            str: The save path of the downloaded file.
        """
        url  = f"{self.url}/download_file"
        payload = {
            'file_path': file_path
        }
        response = requests.post(url, json=payload, timeout=10, cookies=self.cookies,)
        response.raise_for_status()

        save_path = os.path.join(recorder.record_root_dir, file_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return save_path

    def get_workspace_structure(self) -> dict:
        """
        Get the structure of the workspace from the ToolServer.

        Returns:
            dict: The structure of the workspace.
        """
        url  = f"{self.url}/get_workspace_structure"
        response = requests.post(url, timeout=10, cookies=self.cookies,)
        response.raise_for_status()
        response = response.json()
        return response

    def download_all_files(self):
        """
        Download all the files in the workspace from the ToolServer.

        Returns:
            str: The save path of the downloaded workspace.
        """
        url  = f"{self.url}/download_workspace"
        response = requests.post(url, cookies=self.cookies,)
        response.raise_for_status()

        save_path = os.path.join(recorder.record_root_dir, 'workspace.zip')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return save_path

    def get_available_tools(self):
        """
        Get the available tools from the ToolServer.

        Returns:
            The available tools.
        """
        payload = {
        }
        url  = f"{self.url}/get_available_tools"
        cache_output = recorder.query_tool_server_cache(url, payload)
        try:
            if cache_output != None:

                response = cache_output["tool_output"]
                status_code = cache_output["response_status_code"]
            else:
                response = requests.post(url, json=payload, timeout=10, cookies=self.cookies)
                status_code = response.status_code
                response.raise_for_status()
                response = response.json()
                if not isinstance(response, dict):
                    response = json.loads(response)

            recorder.regist_tool_server(url=url,
                                        payload=payload,
                                        tool_output=response,
                                        response_status_code=status_code)
            return response
        except Exception as e:
            raise Exception(f"Error when fetching available tools: {e}")

    def retrieve_rapidapi_tools(self, query: str, top_k: int = 10):
        """
        Retrieve RapidAPI tools from the ToolServer.

        Args:
            query (str): The query for retrieving tools.
            top_k (int, optional): The number of tools to retrieve. Defaults to 10.

        Returns:
            The retrieved tools and the description of the tools in JSON format.
        """
        url  = f"{self.url}/retrieving_tools"
        payload = {
            "question": query,
            "top_k": top_k
        }
        cache_output = recorder.query_tool_server_cache(url, payload)
        try:
            if cache_output != None:
                response = cache_output["tool_output"]
                status_code = cache_output["tool_output_status_code"]
            else:
                response = requests.post(url, json=payload, timeout=20, cookies=self.cookies)
                status_code = response.status_code
                response = response.json()
                if not isinstance(response, dict):
                    response = json.loads(response)
            recorder.regist_tool_server(url=url,
                                        payload=payload,
                                        tool_output=response,
                                        response_status_code=status_code)
            retrieved_tools = response["retrieved_tools"]
            tools_json = response["tools_json"]
            for tool_json in tools_json:
                function_manager.register_function(tool_json)
        except Exception as e:
            logger.typewriter_log(
                "Tool Retrieval Failed, nothing will be retrieved, please fix here.",
                Fore.RED,
            )
            print(f"Error when retrieving tools: {e}")
            print(response)
            return None, None

        return retrieved_tools, tools_json

    def get_json_schema_for_tools(self, command_names):
        """
        Get the JSON schema for the specified tools from the ToolServer.

        Args:
            command_names: The names of the tools.

        Returns:
            The JSON schema for the tools.
        """
        url  = f"{self.url}/get_json_schema_for_tools"
        payload = {
            "tool_names": command_names
        }
        cache_output = recorder.query_tool_server_cache(url, payload)
        try:
            if cache_output != None:
                response = cache_output["tool_output"]
                status_code = cache_output["tool_output_status_code"]
            else:
                response = requests.post(url, json=payload, timeout=10, cookies=self.cookies)
                status_code = response.status_code
                response = response.json()
                if not isinstance(response, dict):
                    try:
                        response = json.loads(response)
                    except:
                        pass
            recorder.regist_tool_server(url=url,
                                        payload=payload,
                                        tool_output=response,
                                        response_status_code=status_code)
            function_manager.register_function(response)
            return response

        except Exception as e:
            print(f"Error when fetching openai function jsons: {e}")
            return None


    # @func_set_timeout()
    def execute_command_client(
            self,
            command_name,
            arguments={},
            #input_hash_id,
        ):
        """
        Execute a command on the ToolServer.

        Args:
            command_name (str): The name of the command.
            arguments (dict, optional): The arguments for the command. Defaults to {}.
            input_hash_id: The hash ID of the input.

        Returns:
            mixed: The result of the command and the tool output status code.
        """
        # return "sorry, the server is not available now", ToolCallStatusCode.SERVER_ERROR, input_hash_id
        url  = f"{self.url}/execute_tool"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except:
                pass
        payload = {
            "tool_name": command_name,
            "arguments": arguments,
            # "hash_id": input_hash_id,
        }

        cache_output = recorder.query_tool_server_cache(url, payload)


        if CONFIG['experiment']['redo_action'] or cache_output is None:
            response = requests.post(url, json=payload, cookies=self.cookies)
            response_status_code = response.status_code
            if response.status_code in {200, 450}:
                command_result = response.json()
                command_result = unwrap_tool_response(command_result)
            else:
                command_result = response.text

        if cache_output != None:
            command_result = cache_output["tool_output"]
            response_status_code = cache_output["response_status_code"]


        recorder.regist_tool_server(url=url,
                                    payload=payload,
                                    tool_output=command_result,
                                    response_status_code=response_status_code)

        # setting tool_output_status_code according to status_code
        if response_status_code == 200:
            tool_output_status_code = ToolCallStatusCode.TOOL_CALL_SUCCESS
        elif response_status_code == 404:
            tool_output_status_code = ToolCallStatusCode.HALLUCINATE_NAME
        elif response_status_code == 422:
            tool_output_status_code = ToolCallStatusCode.FORMAT_ERROR
        elif response_status_code == 450:
            tool_output_status_code = ToolCallStatusCode.TIMEOUT_ERROR
        elif response_status_code == 500:
            tool_output_status_code = ToolCallStatusCode.TOOL_CALL_FAILED
        elif response_status_code == 503:
            tool_output_status_code = ToolCallStatusCode.SERVER_ERROR
            raise Exception(f"Server Error: {command_result}")
        else:
            tool_output_status_code = ToolCallStatusCode.OTHER_ERROR



        return command_result, tool_output_status_code


toolserver_interface = ToolServerInterface()


class FunctionHandler():
    """
    The handler for functions.
    """

    def __init__(self):
        self.subtask_submit_function = function_manager.get_function_schema('subtask_submit')


        # TODO: support more complex versions of human help, like collaborative debugging.
        self.ask_human_for_help_function  = function_manager.get_function_schema('ask_human_for_help')
        self.human_interruption_function = function_manager.get_function_schema('human_interruption')

        self.avaliable_tools_description_list = []

    def log_task_submit(self, arguments):
        """
        Log the task submission.

        Args:
            arguments: The arguments of the task submission.
        """
        logger.typewriter_log(
            "-=-=-=-=-=-=-= SUBTASK SUBMITTED -=-=-=-=-=-=-=", Fore.YELLOW, ""
        )
        logger.typewriter_log(
            "submit_type:", Fore.YELLOW, f"{arguments['submit_type']}"
        )
        logger.typewriter_log(
            "success:", Fore.YELLOW, f"{arguments['result']['success']}"
        )
        logger.typewriter_log(
            "conclusion:", Fore.YELLOW, f"{arguments['result']['conclusion']}"
        )
        if "milestones" in arguments["result"].keys():
            logger.typewriter_log("milestones:", Fore.YELLOW)
            for milestone in arguments["result"]["milestones"]:
                line = milestone.lstrip("- ")
                logger.typewriter_log("- ", Fore.GREEN, line.strip())
        logger.typewriter_log(
            "need_for_plan_refine:",
            Fore.YELLOW,
            f"{arguments['suggestions_for_latter_subtasks_plan']['need_for_plan_refine']}",
        )
        logger.typewriter_log(
            "plan_suggestions:",
            Fore.YELLOW,
            f"{arguments['suggestions_for_latter_subtasks_plan']['reason']}",
        )

    def change_subtask_handle_function_enum(self, function_name_list: List[str]):
        """
        Change the subtask handling function enumeration.

        Args:
            function_name_list: The list of function names.
        """
        match CONFIG.default_request_type:
            case 'openai':
                self.subtask_handle_function = function_manager.get_function_schema('subtask_handle')
                self.subtask_handle_function["parameters"]["properties"]["tool_call"]["properties"]["tool_name"]["enum"] = function_name_list
            case 'xagent':
                pass
            case _:
                raise NotImplementedError(f"Request type {self.config.default_request_type} not implemented")

    def intrinsic_tools(self, enable_ask_human_for_help):
        """
        Get the intrinsic tools.

        Args:
            enable_ask_human_for_help: Whether to enable the ask_human_for_help function.

        Returns:
            The intrinsic tools.
        """
        tools = [self.subtask_submit_function,]
        if enable_ask_human_for_help:
            tools.append(self.ask_human_for_help_function)
        tools.extend(self.avaliable_tools_description_list)
        return tools

    def get_functions(self, config):
        """
        Get the functions.

        Args:
            config: The configuration for the functions.

        Returns:
            The intrinsic tools and the description of the tools.
        """
        output = toolserver_interface.get_available_tools()

        available_tools:list = output['available_tools']
        openai_function_jsons:dict = output['tools_json']

        black_list = set(config.tool_blacklist)
        for item in black_list:
            try:
                available_tools.remove(item)
            except ValueError:
                pass
        openai_function_jsons = [openai_function_json for openai_function_json in openai_function_jsons if openai_function_json['name'] not in black_list]

        self.tool_names = available_tools
        self.change_subtask_handle_function_enum(available_tools)
        self.avaliable_tools_description_list = openai_function_jsons
        for tool_json in openai_function_jsons:
            function_manager.register_function(tool_json)
        return self.intrinsic_tools(config.enable_ask_human_for_help), self.avaliable_tools_description_list

    def long_result_summary(self, command: dict, result):
        """
        Summarize the long result.

        Args:
            command (dict): The command.
            result: The result.

        Returns:
            The summarized result.
        """
        if command['name'] == 'WebEnv_browse_website':
            if not isinstance(result, str):
                result = str(result)
            result = function_manager('parse_web_text', webpage=result[:8096], prompt=command['arguments']['goals_to_browse'])
            result['useful_hyperlinks']  = result['useful_hyperlinks'][:3]
        if command['name'] == 'WebEnv_search_and_browse':
            with ThreadPoolExecutor(max_workers=len(result)) as pool:
                f = [
                    pool.submit(
                        function_manager,
                        'parse_web_text',
                        webpage=ret['page'][:8096],
                        prompt=command['arguments']['goals_to_browse'],
                    )
                    for ret in result
                ]
                for ret, thd in zip(result, f):
                    ret['page'] = thd.result()
                    ret['page']['useful_hyperlinks'] = ret['page']['useful_hyperlinks'][:3]

        return result

    def handle_tool_call(self, node: ToolNode, task_handler):
        """
        Handle the tool call.

        Args:
            node (ToolNode): The tool node.
            task_handler: The task handler.

        Returns:
            The result, tool output status code, whether to refine the plan, and the tools used.
        """
        plan_refine = False
        command_name = node.data["command"]["properties"]["name"]
        arguments = node.data["command"]["properties"]["args"]


        logger.typewriter_log(
            "NEXT ACTION: ",
            Fore.CYAN,
            f"COMMAND: {Fore.CYAN}{command_name}{Style.RESET_ALL}  \n"
            f"ARGUMENTS: \n{Fore.CYAN}{arguments}{Style.RESET_ALL}",
        )

        if command_name == "subtask_submit":
            plan_refine, tool_output_status_code, command_result, = self.handle_subtask_submit(arguments)
        elif command_name == "ask_human_for_help":
            plan_refine, tool_output_status_code, command_result, = self.handle_human_help(arguments)
        elif command_name == "human_interruption":
            assert False, "Never call this function"
        elif command_name == '' or command_name is None:
            command_result = ''
            tool_output_status_code = ToolCallStatusCode.TOOL_CALL_SUCCESS
        else:
            command_result, tool_output_status_code, = toolserver_interface.execute_command_client(
                command_name,
                arguments,
                # input_hash_id,
            )
            # retry to get the result
            MAX_RETRY = 10
            retry_time = 0
            while retry_time < MAX_RETRY and tool_output_status_code == ToolCallStatusCode.TIMEOUT_ERROR and isinstance(command_result['detail'], dict) and 'type' in command_result['detail'] and command_result['detail']['type'] == 'retry':
                time.sleep(3)
                retry_time += 1
                command_result, tool_output_status_code, = toolserver_interface.execute_command_client(
                    command_result['detail']['next_calling'],
                    command_result['detail']['arguments'],
                )


            if tool_output_status_code == ToolCallStatusCode.TIMEOUT_ERROR and retry_time == MAX_RETRY:
                command_result = "Timeout and no content returned! Please check the content you submit!"

        if tool_output_status_code == ToolCallStatusCode.TOOL_CALL_SUCCESS:
            command_result = self.long_result_summary({'name': command_name, 'arguments': arguments}, command_result)

        result = f"Command {command_name} returned: " + f"{command_result}"

        node.data["tool_output"] = command_result
        node.data["tool_status_code"] = tool_output_status_code

        # node.workspace_hash_id = output_hash_id
        if result is not None:
            node.history.add("system", result, "action_result")
            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
        else:
            node.history.add("system", "Unable to execute command", "action_result")
            logger.typewriter_log(
                "SYSTEM: ", Fore.YELLOW, "Unable to execute command"
            )

        if tool_output_status_code == ToolCallStatusCode.TOOL_CALL_SUCCESS:
            color = Fore.GREEN
        elif tool_output_status_code == ToolCallStatusCode.SUBMIT_AS_SUCCESS:
            color = Fore.YELLOW
        elif tool_output_status_code == ToolCallStatusCode.SUBMIT_AS_FAILED:
            color = Fore.BLUE
        else:
            color = Fore.RED

        logger.typewriter_log(
            "TOOL STATUS CODE: ", Fore.YELLOW, f"{color}{tool_output_status_code.name}{Style.RESET_ALL}"
        )

        recorder.regist_tool_call(
            tool_name = command_name,
            tool_input = arguments,
            tool_output = command_result,
            tool_status_code = tool_output_status_code.name,
            thought_data={"thought": node.data["thoughts"], "content": node.data["content"]},
        )

        using_tools = {
            "tool_name": command_name,
            "tool_input": arguments,
            "tool_output": command_result,
            "tool_status_code": tool_output_status_code.name,
            "thought_data": {"thought": node.data["thoughts"], "content": node.data["content"]}
        }

        if tool_output_status_code in [ToolCallStatusCode.SUBMIT_AS_SUCCESS ,ToolCallStatusCode.SUBMIT_AS_FAILED]:
            self.log_task_submit(arguments)

        return result, tool_output_status_code, plan_refine, using_tools

    def handle_subtask_submit(self, arguments):
        """
        Handle the subtask submission.

        Args:
            arguments: The arguments of the subtask submission.

        Returns:
            bool: Whether to refine the plan.
            The tool output status code.
            The result.
        """
        if arguments["result"]["success"]:
            tool_output_status_code = ToolCallStatusCode.SUBMIT_AS_SUCCESS
        else:
            tool_output_status_code = ToolCallStatusCode.SUBMIT_AS_FAILED
        plan_refine = bool(
            arguments["suggestions_for_latter_subtasks_plan"][
                "need_for_plan_refine"
            ]
        )
        answer = {
            "content": f"you have successfully submit the subtask as {arguments['submit_type']}"
        }
        command_result = json.dumps(answer, ensure_ascii=False)

        return plan_refine, tool_output_status_code, command_result

    def handle_human_help(self, arguments):
        """
        Handle the ask for human help.

        Args:
            arguments: The arguments for asking human help.

        Returns:
            bool: Whether to refine the plan.
            The tool output status code.
            The result.
        """
        logger.typewriter_log(
            "ASK For Human Help",
            Fore.RED,
            "You must give some suggestions, please type in your feedback and then press 'Enter' to send and continue the loop"
        )
        url = "ask_human"
        payload = arguments
        tool_cache = recorder.query_tool_server_cache(url=url, payload=payload)
        if tool_cache != None:
            command_result = tool_cache["tool_output"]
            status_code = tool_cache["response_status_code"]
        else:
            human_suggestion = input()
            command_result = json.dumps({"output": f"{human_suggestion}"}, ensure_ascii=False)
            status_code = "human has no status :)"
        recorder.regist_tool_server(
            url=url,
            payload=payload,
            tool_output=command_result,
            response_status_code=status_code,
        )

        plan_refine = False
        return plan_refine, ToolCallStatusCode.TOOL_CALL_SUCCESS, command_result

function_handler = FunctionHandler()