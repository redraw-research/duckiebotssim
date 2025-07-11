// MIT License (c) 2019 BYU PCCL see LICENSE file

#include "OrientationSensor.h"
#include "DuckiebotsSim.h"
#include "Components/PrimitiveComponent.h"
#include "Components/StaticMeshComponent.h"

UOrientationSensor::UOrientationSensor() {
	PrimaryComponentTick.bCanEverTick = true;
	SensorName = "OrientationSensor";
}

void UOrientationSensor::InitializeSensor() {
	Super::InitializeSensor();

	Controller = static_cast<AHolodeckPawnController*>(this->GetAttachmentRootActor()->GetInstigator()->Controller);
	Parent = static_cast<UPrimitiveComponent*>(this->GetAttachParent());
	RootMesh = static_cast<UStaticMeshComponent*>(this->GetAttachParent());

	World = Parent->GetWorld();
}

// Called every frame
void UOrientationSensor::TickSensorComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) {
	if (Parent != nullptr && RootMesh != nullptr) {
		FVector Forward = RootMesh->GetForwardVector();
		FVector Right = RootMesh->GetRightVector();
		FVector Up = RootMesh->GetUpVector();

		float* FloatBuffer = static_cast<float*>(Buffer);
		Forward = ConvertLinearVector(Forward, NoScale);
		FVector Left = ConvertLinearVector(-Right, NoScale);
		Up = ConvertLinearVector(Up, NoScale);

		FloatBuffer[0] = Forward.X;
		FloatBuffer[1] = Forward.Y;
		FloatBuffer[2] = Forward.Z;
		FloatBuffer[3] = Right.X;
		FloatBuffer[4] = Right.Y;
		FloatBuffer[5] = Right.Z;
		FloatBuffer[6] = Up.X;
		FloatBuffer[7] = Up.Y;
		FloatBuffer[8] = Up.Z;
	}
}