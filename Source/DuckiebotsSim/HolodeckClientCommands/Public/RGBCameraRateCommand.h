#pragma once

#include "DuckiebotsSim.h"

#include "Command.h"
#include "RGBCameraRateCommand.generated.h"

/**
* RGBCameraRateCommand
* 
*/
UCLASS(ClassGroup = (Custom))
class DUCKIEBOTSSIM_API URGBCameraRateCommand : public UCommand {
	GENERATED_BODY()
public:
	//See UCommand for the documentation of this overridden function. 
	void Execute() override;
};
