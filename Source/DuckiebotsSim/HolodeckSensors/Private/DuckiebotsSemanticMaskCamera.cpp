// Fill out your copyright notice in the Description page of Project Settings.


#include "DuckiebotsSemanticMaskCamera.h"

#include "Materials/Material.h"


UDuckiebotsSemanticMaskCamera::UDuckiebotsSemanticMaskCamera() {
	SensorName = "DuckiebotsSemanticMaskCamera";
}

void UDuckiebotsSemanticMaskCamera::InitializeSensor()
{
	Super::InitializeSensor();
	// Cast<UMaterial>();
	FWeightedBlendable SemanticMaskMaterial = FWeightedBlendable(1.0, StaticLoadObject(UMaterial::StaticClass(), nullptr, TEXT("/Game/SegmentationPostProcessingMaterial.SegmentationPostProcessingMaterial")));
	SceneCapture->PostProcessSettings.WeightedBlendables.Array.Add(SemanticMaskMaterial);
}
