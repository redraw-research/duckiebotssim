#pragma once

#include "DuckiebotsSim.h"
#include "HolodeckSensor.h"
#include "HolodeckViewportClient.h"
#include "RenderRequest.h"
#include "Components/SceneCaptureComponent2D.h"
#include "HolodeckCamera.generated.h"

/**
* HolodeckCamera
* Abstract base class for cameras within holodeck
* A camera is anything that needs to access visual information.
* Two examples include a depth sensor and a standard camera.
*/
UCLASS(Blueprintable, ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class DUCKIEBOTSSIM_API UHolodeckCamera : public UHolodeckSensor
{
	GENERATED_BODY()

public:
	/**
	* Default Constructor
	*/
	UHolodeckCamera();

	/**
	* InitializeSensor
	* Sets up the class
	*/
	virtual void InitializeSensor() override;

	/**
	* Allows parameters to be set dynamically
	*/
	virtual void ParseSensorParms(FString ParmsJson) override;

protected:
	//Checkout HolodeckSensor.h for the documentation for this overridden function.
	virtual void TickSensorComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction);
	FColor* Buffer;
	FRenderRequest RenderRequest;

	UPROPERTY()
	UTextureRenderTarget2D* TargetTexture;

	UPROPERTY(EditAnywhere)
	USceneCaptureComponent2D* SceneCapture;
	
	UPROPERTY(EditAnywhere)
	int CaptureWidth = 84;

	UPROPERTY(EditAnywhere)
	int CaptureHeight = 84;

private:

	bool bPointerGivenToViewport = false;
	UHolodeckViewportClient* ViewportClient;
};
