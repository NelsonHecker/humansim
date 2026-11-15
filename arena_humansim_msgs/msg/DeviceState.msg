# One assistive device an agent holds, posed from the agent's gait phase this tick, world frame (map), metres
string CANE=cane
string ROLLATOR=rollator
string WALKING_FRAME=walking_frame

int32 agent_id
string kind                    # cane | rollator | walking_frame
string model                   # object model a renderer draws for it (Arena asset name), empty = none
geometry_msgs/Point start      # cane: the fist's grip centre; rollator, walking_frame: the origin, under the middle of the grips (z: lift off the floor)
geometry_msgs/Point end        # cane: the tip; rollator, walking_frame: = start
geometry_msgs/Point centre     # rollator, walking_frame: centre of the footprint on the floor; cane: = end
float64 yaw                    # rollator, walking_frame: heading of the device (the agent's); cane: heading of the shaft, start to end
float64 radius                 # cane: shaft radius; rollator, walking_frame: 0
geometry_msgs/Vector3 size     # rollator, walking_frame: footprint length (along yaw), width, height; cane: 0
