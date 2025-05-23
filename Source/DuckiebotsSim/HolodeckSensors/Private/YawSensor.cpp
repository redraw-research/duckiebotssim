// Fill out your copyright notice in the Description page of Project Settings.


#include "YawSensor.h"
#include "DuckiebotAgent.h"


void UYawSensor::InitializeSensor()
{
	Super::InitializeSensor();
	OwningActor = this->GetAttachmentRootActor();
	PrimaryComponentTick.bCanEverTick = true;
	SensorName = "YawSensor";
}

void UYawSensor::TickSensorComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	//check if your parent pointer is valid, and if the sensor is on. Then get the location and buffer, then send the location to the buffer. 
	if (OwningActor != nullptr && bOn) {
		TObjectPtr<class ADuckiebotAgent> DuckiebotAgentOwner = Cast<ADuckiebotAgent>(OwningActor);
		if (nullptr == DuckiebotAgentOwner)
		{
			UE_LOG(LogDuckiebotsSim, Warning, TEXT("YawSensor doesn't have a DuckiebotAgent owner to communicate with."));
		} else
		{
			float* FloatBuffer = static_cast<float*>(Buffer);
			FloatBuffer[0] = DuckiebotAgentOwner->GetActorRotation().Yaw;
		}
	}
}
