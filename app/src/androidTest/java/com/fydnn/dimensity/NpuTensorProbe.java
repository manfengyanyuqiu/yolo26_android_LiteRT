package com.fydnn.dimensity;

import android.app.Instrumentation;
import android.os.Bundle;
import com.google.ai.edge.litert.*;
import java.io.*;
import java.nio.*;
import org.tensorflow.lite.*;

/** Numerical diagnostic: exact packed input, arbitrary fixed single-output graph. */
final class NpuTensorProbe {
    static void run(Instrumentation test,Bundle args,PrintWriter writer)throws Exception {
        File directory=test.getTargetContext().getExternalFilesDir("diagnostics");
        File file=new File(directory,args.getString("probeModel"));
        DataType inputType,outputType;
        try(Interpreter metadata=new Interpreter(file,new Interpreter.Options().setNumThreads(1))) {
            inputType=metadata.getInputTensor(0).dataType();outputType=metadata.getOutputTensor(0).dataType();
        }
        byte[] input=java.nio.file.Files.readAllBytes(new File(directory,args.getString("probeInput")).toPath());
        try(Environment environment=Environment.create(test.getTargetContext(),new BuiltinNpuAcceleratorProvider(test.getTargetContext()));
            CompiledModel model=NpuOnlyModel.create(file.getAbsolutePath(),environment)) {
            var inputs=model.createInputBuffers(0);var outputs=model.createOutputBuffers(0);
            try {
                if(inputs.size()!=1||outputs.size()!=1)throw new IllegalArgumentException("Probe requires one input/output");
                var in=inputs.get(0);var out=outputs.get(0);
                if(inputType==DataType.FLOAT32) {
                    float[] values=new float[input.length/4];ByteBuffer.wrap(input).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values);in.writeFloat(values);
                }else in.writeInt8(input);
                try(PreparedNpuCall call=PreparedNpuCall.create(model,in,out)) {
                    call.run();byte[] bytes;
                    if(outputType==DataType.FLOAT32) {
                        float[] values=out.readFloat();ByteBuffer raw=ByteBuffer.allocate(values.length*4).order(ByteOrder.LITTLE_ENDIAN);raw.asFloatBuffer().put(values);bytes=raw.array();
                    }else bytes=out.readInt8();
                    try(FileOutputStream destination=new FileOutputStream(new File(directory,"probe-output.bin"))){destination.write(bytes);}
                }
                writer.println("NPU tensor probe PASS");writer.flush();
            }finally{for(var buffer:outputs)buffer.close();for(var buffer:inputs)buffer.close();}
        }
    }
}
