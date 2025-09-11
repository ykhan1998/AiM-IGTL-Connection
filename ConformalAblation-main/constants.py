import getpass
import os

WORK_DIR = f"/home/{getpass.getuser()}/ConformalAblation"

if os.name == 'nt':
    WORK_DIR = f"C:/Users/{os.getlogin()}/Sound/ConformalAblation"

SETTINGS_DIR = WORK_DIR + "/settings"
SAVE_DIR = WORK_DIR + "/save"
LESION_MASK_DIR = WORK_DIR + "/lesion_masks"
ACPR_PATH = WORK_DIR + "ACPR"
MATLAB_FUNC_PATH = WORK_DIR + "/matlab_functions"
TRAJ_DIR = WORK_DIR + "/traj"
FIG_DIR = WORK_DIR + "/fig"
RES_DIR = WORK_DIR + "/results"

# healthy & cem43<thresh GOOD
STATUS_1 =  170
# healthy & cem43>thresh BAD
STATUS_2 = 0
# tumor & cem43<thresh OK
STATUS_3 = 85
# tumor & cem43>thresh GOOD
STATUS_4 = 255

# consider all lesion is burnt if the number of remaining lesion pixels is less than threshold
LESION_THRESH = 3

# Termination causes
TERM_CAUSES = {"Completed_one_circle": 0,
                "Reached_max step": 1,
                "Too_much_damage": 2,
                "All_lesion_burnt": 3}

STATUS_VALUES = {"healthy_ablated": -0.5,
                "lesion_ablated": 1,
                "healthy_unablated": 0,
                "lesion_unablated": 0}
