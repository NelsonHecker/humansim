# One edge of a robot touching an agent: the first (START) or last (END) tick of the contact, world frame (map)
uint8 START=0
uint8 END=1
string BODY=body               # part: the agent's body; otherwise the kind of the device it holds (DeviceState)

uint8 event
string part                    # body | cane | rollator | walking_frame
int32 robot_id
string robot_name
int32 agent_id
string agent_name
string agent_type
geometry_msgs/Point point      # the part's point nearest the robot centre (z: height on a cane, 0 on the floor)
float64 depth                  # robot overlap into the part (m), START: at the first touch, END: the deepest of the contact
float64 duration               # END: seconds since START, START: 0
geometry_msgs/Pose2D robot_pose
geometry_msgs/Pose2D agent_pose
DeviceState device             # the agent's device at this edge, kind empty when it holds none
