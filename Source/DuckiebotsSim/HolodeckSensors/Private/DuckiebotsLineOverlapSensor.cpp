// Fill out your copyright notice in the Description page of Project Settings.


#include "DuckiebotsLineOverlapSensor.h"
#include "Components/BoxComponent.h"
#include "Components/CapsuleComponent.h"


void UDuckiebotsLineOverlapSensor::InitializeSensor()
{
	Super::InitializeSensor();
	TrackedCollisionComponent = this->GetAttachmentRootActor()->FindComponentByClass<UCapsuleComponent>();
	PrimaryComponentTick.bCanEverTick = true;
	SensorName = "DuckiebotsLineOverlapSensor";
	
}

void UDuckiebotsLineOverlapSensor::TickSensorComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	if (bOn) {
		if (!TrackedCollisionComponent)
		{
			UE_LOG(LogDuckiebotsSim, Warning, TEXT("DuckiebotsLineOverlapSensor doesn't have a TrackedCollisionComponent owner to communicate with."));
		} else
		{
			bool* BoolBuffer = static_cast<bool*>(Buffer);
			TArray<AActor*> OverlappingActors;
			TrackedCollisionComponent->GetOverlappingActors(OverlappingActors);
			bool IsOverlapping = false;
			for (AActor* OverlappingActor : OverlappingActors)
			{
				if (OverlappingActor->Tags.Contains("tape"))
				{
					IsOverlapping = true;
					break;
				}
			}
			BoolBuffer[0] = IsOverlapping;
		}
	}
}
