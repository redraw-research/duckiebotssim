
#include "VelocitySensor.h"
#include "DuckiebotsSim.h"

UVelocitySensor::UVelocitySensor() {
	PrimaryComponentTick.bCanEverTick = true;
	SensorName = "VelocitySensor";
}

void UVelocitySensor::InitializeSensor() {
	Super::InitializeSensor();

	//You need to get the pointer to the object the sensor is attached to. 
	Parent = this->GetAttachmentRootActor();
}

void UVelocitySensor::TickSensorComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) {
	//check if your parent pointer is valid, and if the sensor is on. Then get the velocity and buffer, then send the data to it. 
	if (Parent != nullptr && bOn) {
		FVector Velocity = Parent->GetVelocity();
		// UE_LOG(LogDuckiebotsSim, Warning, TEXT("Velocity Sensor Tick: %d reads %s"), GFrameNumber, *Velocity.ToString());
		Velocity = ConvertLinearVector(Velocity, UEToClient);
		float* FloatBuffer = static_cast<float*>(Buffer);
		FloatBuffer[0] = Velocity.X;
		FloatBuffer[1] = Velocity.Y;
		FloatBuffer[2] = Velocity.Z;
	}
}