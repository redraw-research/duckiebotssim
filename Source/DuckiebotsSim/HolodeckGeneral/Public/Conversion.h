#pragma once

#include "DuckiebotsSim.h"

const float UEUnitsPerMeter = 10.0;
const float UEUnitsPerMeterSquared = 100;

enum ConvertType {UEToClient, ClientToUE, NoScale};

FVector ConvertLinearVector(FVector Vector, ConvertType Type);


FVector ConvertAngularVector(FVector Vector, ConvertType Type);

FRotator ConvertAngularVector(FRotator Rotator, ConvertType);


FVector ConvertTorque(FVector Vector, ConvertType Type);

float ConvertClientDistanceToUnreal(float client);

float ConvertUnrealDistanceToClient(float unreal);