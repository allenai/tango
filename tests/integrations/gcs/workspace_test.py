# from tango.common.testing.steps import FloatStep
# from tango.integrations.gcs.workspace import GCSWorkspace
# from tango.step_info import StepState
# from tango.workspace import Workspace
#
#
# def test_from_url(gcs_workspace: str):
#     workspace = Workspace.from_url(f"gs://{gcs_workspace}")
#     assert isinstance(workspace, GCSWorkspace)
#
#
# def test_direct_usage(gcs_workspace: str):
#     workspace = GCSWorkspace(gcs_workspace)
#
#     step = FloatStep(step_name="float", result=1.0)
#     # run = workspace.register_run([step])
#     # assert run.name in workspace.registered_runs()
#     #
#     # assert workspace.step_info(step).state == StepState.INCOMPLETE
#     # workspace.step_starting(step)
#     # assert workspace.step_info(step).state == StepState.RUNNING
#     # workspace.step_finished(step, 1.0)
#     # assert workspace.step_info(step).state == StepState.COMPLETED
#     # assert workspace.step_result_for_run(run.name, "float") == 1.0
