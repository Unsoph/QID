from roboflow import Roboflow
rf = Roboflow(api_key="RVWVJNxL8UnLnu3SEJMW")
project = rf.workspace("shreyas-xxl2f").project("my-first-project-61agt")
version = project.version(4)
dataset = version.download("yolov8")
                