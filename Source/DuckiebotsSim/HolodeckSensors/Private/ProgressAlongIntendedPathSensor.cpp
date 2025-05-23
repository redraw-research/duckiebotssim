// Fill out your copyright notice in the Description page of Project Settings.


#include "ProgressAlongIntendedPathSensor.h"
#include "DuckiebotAgent.h"


void UProgressAlongIntendedPathSensor::InitializeSensor()
{
	Super::InitializeSensor();
	OwningActor = this->GetAttachmentRootActor();
	PrimaryComponentTick.bCanEverTick = true;
	SensorName = "ProgressAlongIntendedPathSensor";
}

void UProgressAlongIntendedPathSensor::TickSensorComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	//check if your parent pointer is valid, and if the sensor is on. Then get the location and buffer, then send the location to the buffer. 
	if (OwningActor != nullptr && bOn) {
		TObjectPtr<class ADuckiebotAgent> DuckiebotAgentOwner = Cast<ADuckiebotAgent>(OwningActor);
		if (nullptr == DuckiebotAgentOwner)
		{
			UE_LOG(LogDuckiebotsSim, Warning, TEXT("DuckiebotsLoopStatusSensor doesn't have a DuckiebotAgent owner to communicate with."));
		} else
		{
			float* FloatBuffer = static_cast<float*>(Buffer);
			FloatBuffer[0] = DuckiebotAgentOwner->ProgressAlongIntendedPath;
			FloatBuffer[1] = DuckiebotAgentOwner->VelocityAlongIntendedPath;
			FloatBuffer[2] = DuckiebotAgentOwner->DistanceFromIntendedPath;
		}
	}
}
