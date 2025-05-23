// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "HolodeckSensor.h"
#include "DuckiebotsLoopStatusSensor.generated.h"


UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class DUCKIEBOTSSIM_API UDuckiebotsLoopStatusSensor : public UHolodeckSensor
{
	GENERATED_BODY()

public:
	virtual void InitializeSensor() override;

protected:
	virtual int GetNumItems() override { return 4; }; // CollidedWithWhiteLine, CollidedWithYellowLine, IsYellowLineToLeftAndWhiteLineToRight, EnteredNewRoadTile
	virtual int GetItemSize() override { return sizeof(bool); };
	
	virtual void TickSensorComponent(float DeltaTime, ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;
private:
	AActor* OwningActor;
};
