// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "DuckiebotAgent.h"
#include "HolodeckSensor.h"
#include "DuckiebotsLineOverlapSensor.generated.h"


UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class DUCKIEBOTSSIM_API UDuckiebotsLineOverlapSensor : public UHolodeckSensor
{
	GENERATED_BODY()

public:
	virtual void InitializeSensor() override;

protected:
	virtual int GetNumItems() override { return 1; };
	virtual int GetItemSize() override { return sizeof(bool); };
	
	virtual void TickSensorComponent(float DeltaTime, ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;

private:
	TObjectPtr<class UCapsuleComponent> TrackedCollisionComponent;
};
