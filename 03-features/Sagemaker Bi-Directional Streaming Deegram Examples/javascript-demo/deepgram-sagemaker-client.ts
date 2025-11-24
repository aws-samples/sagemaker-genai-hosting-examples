import * as fs from "fs";
import * as path from "path";
import { RequestStreamEvent } from '@aws-sdk/client-sagemaker-runtime-http2';

function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function* streamWavFile(filePath: string): AsyncIterable<RequestStreamEvent> {
    const full = path.resolve(filePath);

    if (!fs.existsSync(full)) {
        throw new Error(`Audio file not found: ${full}`);
    }

    console.log(`Streaming audio: ${full}`);

    const readStream = fs.createReadStream(full, { highWaterMark: 512_000 }); // 512 KB

    for await (const chunk of readStream) {
        yield {
            PayloadPart: {
                Bytes: chunk,
                DataType: "BINARY"
            }
        };
    }
    
    // Keep the stream alive to receive transcription responses after whole audio file is sent
    console.log("Audio sent, waiting for transcription to finish..."); 
    await sleep(15000); // Wait 15 seconds for processing final audio chunk.
    //Long audio files may require sending keep alive packets while the transcript is being processed. see https://developers.deepgram.com/docs/audio-keep-alive for more information.

    // Tell the container we're done
    yield {
        PayloadPart: {
            Bytes: new TextEncoder().encode('{"type":"CloseStream"}'),
            DataType: "UTF8"
        }
    };
}

import {
    SageMakerRuntimeHTTP2Client,
    InvokeEndpointWithBidirectionalStreamCommand
} from '@aws-sdk/client-sagemaker-runtime-http2';

const region = "us-east-1";              // AWS region
const endpointName = "deepgram-nova3-endpoint";        // Your SageMaker Deepgram endpoint name
const audioFile = "path/to/your/audio.wav";            // Local audio file - UPDATE THIS PATH

// Deepgram WebSocket path inside the model container
const modelInvocationPath = "v1/listen";
const modelQueryString = "model=nova-3";

const client = new SageMakerRuntimeHTTP2Client({
    region
});

async function run() {
    console.log("Sending audio to Deepgram via SageMaker...");

    const command = new InvokeEndpointWithBidirectionalStreamCommand({
        EndpointName: endpointName,
        Body: streamWavFile(audioFile),
        ModelInvocationPath: modelInvocationPath,
        ModelQueryString: modelQueryString
    });

    const response = await client.send(command);

    if (!response.Body) {
        console.log("No streaming response received.");
        return;
    }

    const decoder = new TextDecoder();

    for await (const msg of response.Body) {
        if (msg.PayloadPart?.Bytes) {
            const text = decoder.decode(msg.PayloadPart.Bytes);

            try {
                const parsed = JSON.parse(text);
                
                // Extract and display the transcript
                if (parsed.channel?.alternatives?.[0]?.transcript) {
                    const transcript = parsed.channel.alternatives[0].transcript;
                    if (transcript.trim()) {
                        console.log("Transcript:", transcript);
                    }
                }
                    
                // console.debug("Deepgram (raw):", parsed);

            } catch {
                console.error("Deepgram (error):", text);
            }
        }
    }

    console.log("Streaming finished.");
}

run().catch(console.error);
