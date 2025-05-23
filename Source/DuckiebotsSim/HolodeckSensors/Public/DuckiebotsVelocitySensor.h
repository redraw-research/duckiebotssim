// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "DuckiebotAgent.h"
#include "DuckiebotPawnMovement.h"
#include "HolodeckSensor.h"
#include "DuckiebotsVelocitySensor.generated.h"


UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class DUCKIEBOTSSIM_API UDuckiebotsVelocitySensor : public UHolodeckSensor
{
	GENERATED_BODY()

public:
	virtual void InitializeSensor() override;

protected:
	virtual int GetNumItems() override { return 2; }; // forward_velocity, yaw_velocity
	virtual int GetItemSize() override { return sizeof(float); };
	
	virtual void TickSensorComponent(float DeltaTime, ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;

private:
	TObjectPtr<class ADuckiebotAgent> DuckiebotAgentOwner;
	TObjectPtr<class UDuckiebotPawnMovement> TrackedDuckiebotPawnMovement;
};
