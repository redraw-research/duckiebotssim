#pragma once

#include "DuckiebotsSim.h"

#include <map>
// #include "CollisionSensor.h"
// #include "IMUSensor.h"
// #include "JointRotationSensor.h"
// #include "LocationSensor.h"
// #include "OrientationSensor.h"
// #include "PressureSensor.h"
// #include "RelativeSkeletalPositionSensor.h"
#include "RGBCamera.h"
// #include "RotationSensor.h"
// #include "VelocitySensor.h"
// #include "WorldNumSensor.h"
// #include "AbuseSensor.h"
// #include "RangeFinderSensor.h"
#include "ViewportCapture.h"
// #include "DistanceTask.h"
// #include "LocationTask.h"
// #include "FollowTask.h"
// #include "CupGameTask.h"
// #include "CleanUpTask.h"
#include "DuckiebotsLoopStatusSensor.h"
#include "DuckiebotsSemanticMaskCamera.h"

#include "Command.h"
#include "AddSensorCommand.generated.h"

/**
* AddSensorCommand
* Command used to add a sensor to an agent
* Warning: This command is meant for initialization. Adding a sensor with the same name as a previously 
* removed sensor may cause problems.
*
* StringParameters expects five arguments: the agent name, sensor name, sensor class, sensor parameters, and socket.
* NumberParameters expects six arguments: locations x, y, and z and rotations pitch, yaw, and roll.
*/
UCLASS()
class DUCKIEBOTSSIM_API UAddSensorCommand : public UCommand
{
	GENERATED_BODY()

	typedef std::map<FString, UClass*> USensorMapType;

public:
	//See UCommand for the documentation of this overridden function.
	void Execute() override;

};
