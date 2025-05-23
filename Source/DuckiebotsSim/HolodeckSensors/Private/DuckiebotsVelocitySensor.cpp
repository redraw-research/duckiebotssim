// Fill out your copyright notice in the Description page of Project Settings.


#include "DuckiebotsVelocitySensor.h"
#include "DuckiebotAgent.h"


void UDuckiebotsVelocitySensor::InitializeSensor()
{
	Super::InitializeSensor();
	DuckiebotAgentOwner = Cast<ADuckiebotAgent>(this->GetAttachmentRootActor());
	TrackedDuckiebotPawnMovement = Cast<UDuckiebotPawnMovement>(DuckiebotAgentOwner->GetMovementComponent());
	PrimaryComponentTick.bCanEverTick = true;
	SensorName = "DuckiebotsVelocitySensor";
	
}

void UDuckiebotsVelocitySensor::TickSensorComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	//check if your parent pointer is valid, and if the sensor is on. Then get the location and buffer, then send the location to the buffer. 
	if (bOn) {
		if (nullptr == DuckiebotAgentOwner)
		{
			UE_LOG(LogDuckiebotsSim, Warning, TEXT("DuckiebotsVelocitySensor doesn't have a DuckiebotAgent owner to communicate with."));
		} else
		{
			float* FloatBuffer = static_cast<float*>(Buffer);
			FloatBuffer[0] = ConvertUnrealDistanceToClient(TrackedDuckiebotPawnMovement->GetCurrentForwardVelocity());
			FloatBuffer[1] = TrackedDuckiebotPawnMovement->GetCurrentYawVelocityFromInputs();
		}
	}
}
