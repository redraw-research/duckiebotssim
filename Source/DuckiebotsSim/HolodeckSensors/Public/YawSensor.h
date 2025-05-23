// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "HolodeckSensor.h"
#include "YawSensor.generated.h"


UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class DUCKIEBOTSSIM_API UYawSensor : public UHolodeckSensor
{
	GENERATED_BODY()

public:
	virtual void InitializeSensor() override;

protected:
	virtual int GetNumItems() override { return 1; };
	virtual int GetItemSize() override { return sizeof(float); };
	
	virtual void TickSensorComponent(float DeltaTime, ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;

private:
	AActor* OwningActor;
};
