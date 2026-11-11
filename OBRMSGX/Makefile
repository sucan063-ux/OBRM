SGX_SDK ?= /opt/intel/sgxsdk
SGX_MODE ?= SIM
SGX_ARCH ?= x64
SGX_DEBUG ?= 0

include $(SGX_SDK)/buildenv.mk

ifeq ($(shell getconf LONG_BIT), 32)
	SGX_ARCH := x86
endif

ifeq ($(SGX_ARCH), x86)
	SGX_COMMON_FLAGS := -m32
	SGX_LIBRARY_PATH := $(SGX_SDK)/lib
	SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x86/sgx_sign
	SGX_EDGER8R := $(SGX_SDK)/bin/x86/sgx_edger8r
else
	SGX_COMMON_FLAGS := -m64
	SGX_LIBRARY_PATH := $(SGX_SDK)/lib64
	SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x64/sgx_sign
	SGX_EDGER8R := $(SGX_SDK)/bin/x64/sgx_edger8r
endif

ifeq ($(SGX_DEBUG), 1)
	SGX_COMMON_FLAGS += -O0 -g
else
	SGX_COMMON_FLAGS += -O2
endif

SGX_COMMON_FLAGS += -Wall -Wextra -Wno-unused-parameter
SGX_COMMON_CFLAGS := $(SGX_COMMON_FLAGS)
SGX_COMMON_CXXFLAGS := $(SGX_COMMON_FLAGS) -std=c++11

ifneq ($(SGX_MODE), HW)
	Urts_Library_Name := sgx_urts_sim
	Uae_Service_Library_Name := sgx_uae_service_sim
	Trts_Library_Name := sgx_trts_sim
	Service_Library_Name := sgx_tservice_sim
else
	Urts_Library_Name := sgx_urts
	Uae_Service_Library_Name := sgx_uae_service
	Trts_Library_Name := sgx_trts
	Service_Library_Name := sgx_tservice
endif

App_Name := app
Enclave_Name := enclave.so
Signed_Enclave_Name := enclave.signed.so
Enclave_Config_File := Enclave/Enclave.config.xml
Enclave_Test_Key := Enclave/Enclave_private_test.pem

App_Cpp_Files := App/App.cpp
App_Include_Paths := -IApp -ICommon -I$(SGX_SDK)/include
App_Cpp_Flags := -fPIC -Wno-attributes $(App_Include_Paths)
ifneq ($(SGX_MODE), HW)
App_Cpp_Flags += -DOBRM_SGX_SIMULATION=1
endif
App_Link_Flags := -L$(SGX_LIBRARY_PATH) -Wl,-rpath-link,$(SGX_LIBRARY_PATH) -l$(Urts_Library_Name) -l$(Uae_Service_Library_Name) -lpthread
App_Cpp_Objects := $(App_Cpp_Files:.cpp=.o)

Enclave_Cpp_Files := Enclave/Enclave.cpp
Enclave_Include_Paths := -IEnclave -ICommon -I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc -I$(SGX_SDK)/include/libcxx
Enclave_C_Flags := $(Enclave_Include_Paths) -nostdinc -fvisibility=hidden -fpie -ffunction-sections -fdata-sections $(MITIGATION_CFLAGS)
Enclave_Cpp_Flags := $(Enclave_C_Flags) -nostdinc++
Enclave_Link_Flags := $(MITIGATION_LDFLAGS) -Wl,-z,relro,-z,now,-z,noexecstack \
	-Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L$(SGX_TRUSTED_LIBRARY_PATH) \
	-Wl,--whole-archive -l$(Trts_Library_Name) -Wl,--no-whole-archive \
	-Wl,--start-group -lsgx_tstdc -lsgx_tcxx -lsgx_tcrypto -l$(Service_Library_Name) -Wl,--end-group \
	-Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined \
	-Wl,-pie,-eenclave_entry -Wl,--export-dynamic \
	-Wl,--defsym,__ImageBase=0 -Wl,--gc-sections \
	-Wl,--version-script=Enclave/Enclave.lds
Enclave_Cpp_Objects := $(Enclave_Cpp_Files:.cpp=.o)

.PHONY: all clean

all: $(App_Name) $(Signed_Enclave_Name)
	@echo "Build complete: SGX_MODE=$(SGX_MODE)"

App/Enclave_u.h: Enclave/Enclave.edl
	@cd App && $(SGX_EDGER8R) --untrusted ../Enclave/Enclave.edl --search-path ../Enclave --search-path $(SGX_SDK)/include
	@echo "GEN  =>  $@"

App/Enclave_u.c: App/Enclave_u.h

App/Enclave_u.o: App/Enclave_u.c App/Enclave_u.h
	@$(CC) $(SGX_COMMON_CFLAGS) $(App_Cpp_Flags) -c $< -o $@
	@echo "CC   <=  $<"

App/%.o: App/%.cpp App/Enclave_u.h
	@$(CXX) $(SGX_COMMON_CXXFLAGS) $(App_Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"

$(App_Name): App/Enclave_u.o $(App_Cpp_Objects)
	@$(CXX) $^ -o $@ $(App_Link_Flags)
	@echo "LINK =>  $@"

Enclave/Enclave_t.h: Enclave/Enclave.edl
	@cd Enclave && $(SGX_EDGER8R) --trusted ../Enclave/Enclave.edl --search-path ../Enclave --search-path $(SGX_SDK)/include
	@echo "GEN  =>  $@"

Enclave/Enclave_t.c: Enclave/Enclave_t.h

Enclave/Enclave_t.o: Enclave/Enclave_t.c Enclave/Enclave_t.h
	@$(CC) $(SGX_COMMON_CFLAGS) $(Enclave_C_Flags) -c $< -o $@
	@echo "CC   <=  $<"

Enclave/%.o: Enclave/%.cpp Enclave/Enclave_t.h
	@$(CXX) $(SGX_COMMON_CXXFLAGS) $(Enclave_Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"

Enclave/Enclave.o: Common/ObrmSgxTypes.h Common/VectorFormat.h
App/App.o: Common/ObrmSgxTypes.h Common/VectorFormat.h

$(Enclave_Name): Enclave/Enclave_t.o $(Enclave_Cpp_Objects)
	@$(CXX) $^ -o $@ $(Enclave_Link_Flags)
	@echo "LINK =>  $@"

$(Signed_Enclave_Name): $(Enclave_Name)
ifeq ($(wildcard $(Enclave_Test_Key)),)
	@openssl genrsa -out $(Enclave_Test_Key) -3 3072
endif
	@$(SGX_ENCLAVE_SIGNER) sign -key $(Enclave_Test_Key) -enclave $(Enclave_Name) -out $@ -config $(Enclave_Config_File)
	@echo "SIGN =>  $@"

clean:
	@rm -f $(App_Name) $(Enclave_Name) $(Signed_Enclave_Name) App/Enclave_u.* Enclave/Enclave_t.* \
		$(App_Cpp_Objects) App/Enclave_u.o $(Enclave_Cpp_Objects) Enclave/Enclave_t.o $(Enclave_Test_Key)
