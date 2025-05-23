// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "GameFramework/PawnMovementComponent.h"
#include "DuckiebotPawnMovement.generated.h"

/**
 * 
 */
UCLASS(ClassGroup = Movement, meta = (BlueprintSpawnableComponent))
class DUCKIEBOTSSIM_API UDuckiebotPawnMovement : public UPawnMovementComponent
{
	//GENERATED_BODY()
	GENERATED_UCLASS_BODY()

	//Begin UActorComponent Interface
	virtual void TickComponent(float DeltaTime, enum ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;
	//End UActorComponent Interface

	//Begin UMovementComponent Interface
	virtual float GetMaxSpeed() const override { return MaxSpeed; }

protected:
	virtual bool ResolvePenetrationImpl(const FVector& Adjustment, const FHitResult& Hit, const FQuat& NewRotation) override;
public:
	//End UMovementComponent Interface

	/** Deceleration applied when there is no input (rate of change of velocity) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = DuckiebotPawnMovement)
	bool bUseSimplePhysics;
	
	/** Maximum velocity magnitude allowed for the controlled Pawn. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = DuckiebotPawnMovement)
	float MaxSpeed;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = DuckiebotPawnMovement)
	float ReverseSpeedCoefficient;
	
	/** Acceleration applied by input (rate of change of velocity) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = DuckiebotPawnMovement)
	float Acceleration;

	/** Deceleration applied when there is no input (rate of change of velocity) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = DuckiebotPawnMovement)
	float Deceleration;

	/**
	 * Setting affecting extra force applied when changing direction, making turns have less drift and become more responsive.
	 * Velocity magnitude is not allowed to increase, that only happens due to normal acceleration. It may decrease with large direction changes.
	 * Larger values apply extra force to reach the target direction more quickly, while a zero value disables any extra turn force.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = DuckiebotPawnMovement, meta = (ClampMin = "0", UIMin = "0"))
	float TurningBoost;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = DuckiebotPawnMovement, meta = (ClampMin = "0", UIMin = "0", ClampMax = "1", UIMax = "1"))
	float TurningDeadzone;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = DuckiebotPawnMovement, meta = (ClampMin = "0", UIMin = "0", ClampMax = "200", UIMax = "200"))
	float TurningScale;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = DuckiebotPawnMovement, meta = (ClampMin = "0", UIMin = "0", ClampMax = "1", UIMax = "1"))
	float VelDeadzone;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = DuckiebotPawnMovement, meta = (ClampMin = "0", UIMin = "0", ClampMax = "10", UIMax = "10"))
	float VelScale;

	// Begin Custom DuckiebotPawn Interface
	float YawInput;
	float VelInput;

	UFUNCTION(BlueprintCallable, Category = "Game|Player", meta = (Keywords = "left right turn"))
	virtual void SetYawInput(float Val);

	UFUNCTION(BlueprintCallable, Category = "Game|Player", meta = (Keywords = "forward accel gas"))
	virtual void SetVelInput(float Val);

	float GetCurrentYawVelocityFromInputs();
	float GetCurrentForwardVelocity();

protected:
	float YawVelocityFromInputs;

	/** Update Velocity based on input. Also applies gravity. */
	virtual void ApplyControlInputToVelocity(float DeltaTime);

	/** Prevent Pawn from leaving the world bounds (if that restriction is enabled in WorldSettings) */
	virtual bool LimitWorldBounds();

	/** Set to true when a position correction is applied. Used to avoid recalculating velocity when this occurs. */
	UPROPERTY(Transient)
	uint32 bPositionCorrected : 1;
	
};
