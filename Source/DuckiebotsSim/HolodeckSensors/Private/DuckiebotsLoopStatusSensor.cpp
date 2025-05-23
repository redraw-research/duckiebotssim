// Fill out your copyright notice in the Description page of Project Settings.


#include "DuckiebotsLoopStatusSensor.h"

#include "DuckiebotAgent.h"


void UDuckiebotsLoopStatusSensor::InitializeSensor()
{
	Super::InitializeSensor();
	OwningActor = this->GetAttachmentRootActor();
	SensorName = "DuckiebotsLoopStatusSensor";
}

void UDuckiebotsLoopStatusSensor::TickSensorComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	//check if your parent pointer is valid, and if the sensor is on. Then get the location and buffer, then send the location to the buffer. 
	if (OwningActor != nullptr && bOn) {
		TObjectPtr<class ADuckiebotAgent> DuckiebotAgentOwner = Cast<ADuckiebotAgent>(OwningActor);
		if (nullptr == DuckiebotAgentOwner)
		{
			UE_LOG(LogDuckiebotsSim, Warning, TEXT("DuckiebotsLoopStatusSensor doesn't have a DuckiebotAgent owner to communicate with."));
		} else
		{
			DuckiebotAgentOwner->UpdateDuckiebotLoopStatusBools();
			bool* BoolBuffer = static_cast<bool*>(Buffer);
			// CollidedWithWhiteLine, CollidedWithYellowLine, IsYellowLineToLeftAndWhiteLineToRight, EnteredNewRoadTile
			BoolBuffer[0] = DuckiebotAgentOwner->CollidedWithWhiteLine;
			BoolBuffer[1] = DuckiebotAgentOwner->CollidedWithYellowLine;
			BoolBuffer[2] = DuckiebotAgentOwner->IsYellowLineToLeftAndWhiteLineToRight;
			BoolBuffer[3] = DuckiebotAgentOwner->EnteredNewRoadTile;
			DuckiebotAgentOwner->ClearTemporaryDuckiebotLoopStatusBools();
		}
	}
}
