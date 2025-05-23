// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "RGBCamera.h"
#include "DuckiebotsSemanticMaskCamera.generated.h"


UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class DUCKIEBOTSSIM_API UDuckiebotsSemanticMaskCamera : public URGBCamera
{
	GENERATED_BODY()

public:
	UDuckiebotsSemanticMaskCamera();
	
	virtual void InitializeSensor() override;
};
