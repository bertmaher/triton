#include "./RegisterTritonDialects.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <sstream>
#include <unordered_map>

using namespace llvm;
using namespace mlir;

// Command line options
static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool>
    verifyModule("verify",
                 cl::desc("Verify the module after parsing"),
                 cl::init(true));

static cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialects",
    cl::desc("Allow operation with no registered dialects"), cl::init(false));

// CUDA code generator class
class CUDAGenerator {
public:
  std::stringstream codeStream;

private:
  std::unordered_map<Value*, std::string> valueNames;
  int nextVarId = 0;
  
  std::string getValueName(Value value) {
    Value* valuePtr = &value;
    auto it = valueNames.find(valuePtr);
    if (it != valueNames.end()) {
      return it->second;
    }
    
    std::string name = "var" + std::to_string(nextVarId++);
    valueNames[valuePtr] = name;
    return name;
  }
  
  std::string getAxisName(int axis) {
    switch (axis) {
      case 0: return "x";
      case 1: return "y"; 
      case 2: return "z";
      default: return "x"; // fallback
    }
  }

public:
  void generateGetProgramId(triton::GetProgramIdOp op) {
    std::string resultVar = getValueName(op.getResult());
    std::string axis = getAxisName(op.getAxisAsInt());
    
    codeStream << "  int32_t " << resultVar << " = blockIdx." << axis << ";\n";
  }
  
  void generateConstant(arith::ConstantOp op) {
    std::string resultVar = getValueName(op.getResult());
    
    if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue())) {
      codeStream << "  int32_t " << resultVar << " = " << intAttr.getInt() << ";\n";
    } else {
      codeStream << "  // TODO: Handle non-integer constant\n";
    }
  }
  
  void generateArithMuli(arith::MulIOp op) {
    std::string resultVar = getValueName(op.getResult());
    std::string lhs = getValueName(op.getLhs());
    std::string rhs = getValueName(op.getRhs());
    
    codeStream << "  int32_t " << resultVar << " = " << lhs << " * " << rhs << ";\n";
  }

  std::string getGeneratedCode() {
    return codeStream.str();
  }
  
  void reset() {
    codeStream.str("");
    codeStream.clear();
    valueNames.clear();
    nextVarId = 0;
  }
};

// Operation visitor for conversion
class TritonToCUDAConverter {
private:
  CUDAGenerator &generator;

public:
  TritonToCUDAConverter(CUDAGenerator &gen) : generator(gen) {}
  
  LogicalResult visitOperation(Operation *op) {
    if (auto getProgramIdOp = dyn_cast<triton::GetProgramIdOp>(op)) {
      generator.generateGetProgramId(getProgramIdOp);
      return success();
    }
    
    if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
      generator.generateConstant(constantOp);
      return success();
    }
    
    if (auto muliOp = dyn_cast<arith::MulIOp>(op)) {
      generator.generateArithMuli(muliOp);
      return success();
    }
    
    // For now, just add a comment for unhandled operations
    generator.codeStream << "  // TODO: Handle " << op->getName() << "\n";
    return success();
  }
};

// Main conversion function
static LogicalResult convertTritonGPUToCUDA(ModuleOp module,
                                            raw_ostream &output) {
  output << "// Generated CUDA code from TritonGPU IR\n";
  output << "#include <cuda_runtime.h>\n\n";

  CUDAGenerator generator;
  TritonToCUDAConverter converter(generator);

  // Walk through all operations in the module
  WalkResult result = module.walk([&](Operation *op) {
    // Focus on tt.func operations (Triton kernels)
    if (auto funcOp = dyn_cast<triton::FuncOp>(op)) {
      output << "__global__ void " << funcOp.getName() << "(";
      
      // Generate function parameters
      bool first = true;
      for (auto arg : funcOp.getArguments()) {
        if (!first) output << ", ";
        first = false;
        
        // Simplified parameter handling for now
        if (isa<triton::PointerType>(arg.getType())) {
          output << "float* arg" << arg.getArgNumber();
        } else {
          output << "int32_t arg" << arg.getArgNumber(); 
        }
      }
      
      output << ") {\n";
      
      // Convert function body
      generator.reset();
      funcOp.walk([&](Operation *bodyOp) {
        if (bodyOp != funcOp.getOperation()) {
          converter.visitOperation(bodyOp);
        }
        return WalkResult::advance();
      });
      
      output << generator.getGeneratedCode();
      output << "}\n\n";
    }
    
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }

  // Print original IR as comment for debugging
  output << "/*\nOriginal TritonGPU IR:\n";
  module.print(output);
  output << "\n*/\n";

  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Register command line options
  cl::ParseCommandLineOptions(argc, argv, "Triton to CUDA converter\n");

  // Set up MLIR context with Triton dialects
  MLIRContext context;
  DialectRegistry registry;
  registerTritonDialects(registry);
  context.appendDialectRegistry(registry);
  context.getOrLoadDialect<triton::TritonDialect>();
  context.getOrLoadDialect<triton::gpu::TritonGPUDialect>();

  if (allowUnregisteredDialects)
    context.allowUnregisteredDialects();

  // Set up source manager for error reporting
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  // Open input file
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Add the input file to the source manager
  sourceMgr.AddNewSourceBuffer(std::move(input), SMLoc());

  // Parse the input file
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse input file\n";
    return 1;
  }

  // Verify the module if requested
  if (verifyModule) {
    if (failed(verify(*module))) {
      llvm::errs() << "Module verification failed\n";
      return 1;
    }
  }

  // Open output file
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Convert TritonGPU IR to CUDA
  if (failed(convertTritonGPUToCUDA(*module, output->os()))) {
    llvm::errs() << "Failed to convert TritonGPU IR to CUDA\n";
    return 1;
  }

  // Keep the output file
  output->keep();
  return 0;
}
