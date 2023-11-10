import json
import os
import traceback

from colorama import Fore

from XAgentServer.envs import XAgentServerEnv
from XAgentServer.interaction import XAgentInteraction
from XAgentServer.loggers.logs import Logger
# from XAgentServer.manager import manager
from XAgentServer.response_body import WebsocketResponseBody


class XAgentServer:
    def __init__(self) -> None:
        """Initialization for XAgentServer class.
        
        Attributes:
            logger (Logger): An instance of Logger class.
        """
        self.logger: Logger = None

    def set_logger(self, logger):
        """Set logger for the XAgentServer.
        
        Args:
            logger (Logger): An instance of Logger class.
        """
        self.logger = logger

    async def interact(self, interaction: XAgentInteraction):
        """Interact with the XAgent by executing the required tasks, handling the exceptions and logging the details.
        
        Args:
            interaction (XAgentInteraction): An instance of XAgentInteraction class containing interaction details.
            
        Raises:
            Exception: An error occurred accessing the recorder_root_dir.
            Exception: An error occurred during task execution.
        """
      
        # additional imports are placed inside the function to avoid circular dependencies issues
        from XAgent.agent import (PlanGenerateAgent, PlanRefineAgent,
                                  ReflectAgent, ToolAgent)
        from XAgent.config import CONFIG as config
        from XAgent.global_vars import agent_dispatcher, config
        from XAgent.running_recorder import recorder
        from XAgent.tool_call_handle import (function_handler,
                                             toolserver_interface)
        from XAgent.workflow.base_query import AutoGPTQuery
        from XAgent.workflow.task_handler import TaskHandler
        from XAgent.workflow.working_memory import WorkingMemoryAgent

        config.reload()

        args = interaction.parameter.args
        interaction.base.recorder_root_dir = config.record_dir

        if interaction.base.recorder_root_dir:
            # Check if recorder root directory exists
            if not os.path.exists(interaction.base.recorder_root_dir):
                raise Exception(
                    f"recorder_root_dir {interaction.base.recorder_root_dir} not exists")

            # Load the recorder from disk and get the query
            recorder.load_from_disk(interaction.base.recorder_root_dir)
            query = recorder.get_query()
            self.logger.info(
                f"server is running, the start recorder_root_dir is {interaction.base.recorder_root_dir}")
        else:
            # Create new AutoGPTQuery
            query = AutoGPTQuery(
                role_name=args.get('role_name', 'Assistant'),
                task=args.get('goal', ''),
                plan=args.get('plan', [
                ]),
            )

            self.logger.info(f"server is running, the start query is {args.get('goal', '')}")

        # Register config and query to the recorder
        recorder.regist_query(query)
        recorder.regist_config(config)

        self.logger.info(json.dumps(config.to_dict(), indent=2))
        self.logger.typewriter_log(
            "Human-In-The-Loop",
            Fore.RED,
            str(config.enable_ask_human_for_help),
        )

        # Initialize toolserver interface with the config
        toolserver_interface.lazy_init(config=config)

        # Get the working memory function for communication between different agents
        working_memory_function = WorkingMemoryAgent.get_working_memory_function()
        subtask_functions, tool_functions_description_list = function_handler.get_functions(
            config)

        all_functions = subtask_functions + working_memory_function

        avaliable_agents = [
            PlanGenerateAgent,
            PlanRefineAgent,
            ToolAgent,
            ReflectAgent,
        ]

        for agent in avaliable_agents:
            agent_dispatcher.regist_agent(agent)

        upload_files = args.get("file_list", [])
        if upload_files is not None:
            upload_files = [os.path.join(XAgentServerEnv.Upload.upload_dir, interaction.base.user_id, file) for file in upload_files]
            for file_path in upload_files:
                try:
                    toolserver_interface.upload_file(file_path)
                except Exception as e:
                    self.logger.typewriter_log(
                        "Error happens when uploading file",
                        Fore.RED,
                        f"{file_path}\n{e}",
                    )

        task_handler = TaskHandler(
            config=config,
            query=query,
            interaction=interaction,
            function_list=all_functions,
            tool_functions_description_list=tool_functions_description_list,
        )

        try:
            self.logger.info("Start outer loop async")
            await task_handler.outer_loop_async()
        except Exception as e:
            self.logger.info(traceback.format_exc())
            raise e
        finally:
            toolserver_interface.download_all_files()
            toolserver_interface.close()