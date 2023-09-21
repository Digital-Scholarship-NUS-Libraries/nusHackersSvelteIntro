<script lang="ts">
  import { onMount } from "svelte";
  import { tweened } from "svelte/motion";
  import { cubicInOut } from "svelte/easing";
  import * as tf from "@tensorflow/tfjs";

  import { LayerCake, Svg } from "layercake";
  import { interpolateCubehelix } from "d3-interpolate";

  // adapted from https://layercake.graphics/example/Line
  import Line from "./lib/Line.svelte";
  import Area from "./lib/Area.svelte";
  import AxisY from "./lib/AxisY.svelte";

  const xKey = "myX";
  const yKey = "myY";

  const cubehelixInterpolator = interpolateCubehelix("midnightblue", "orange");

  let disableInput = true;
  let textValue = "";
  let trimmed: string[] = [];
  let sentiment = 0.5;
  const tweenedSentiment = tweened(sentiment, {
    duration: 400,
    easing: cubicInOut,
  });
  let sentimentHistory: number[] = [];
  let data: { myX: number; myY: number }[] = [];

  // adapted from https://observablehq.com/@jashkenas/sentiment-analysis-with-tensorflow-js
  let metadata;
  let model: tf.LayersModel;
  let predict = (_: string[]) => 0.0;

  $: {
    trimmed = textValue
      .trim()
      .toLowerCase()
      .replace(/(\.|\,|\!)/g, "")
      .split(" ");
    sentiment = predict(trimmed);
    tweenedSentiment.set(sentiment);
    sentimentHistory.push(sentiment);
    sentimentHistory = sentimentHistory.slice(-100);
    data = sentimentHistory.map((d, i) => {
      return { myX: i, myY: d };
    });
  }

  onMount(async () => {
    const response = await fetch(
      "https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json"
    );

    metadata = await response.json();

    model = await tf.loadLayersModel(
      "https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json"
    );

    predict = (tokens: string[]) => {
      const inputBuffer = tf.buffer([1, metadata.max_len], "float32");
      tokens.forEach((word, i) =>
        inputBuffer.set(metadata.word_index[word] + metadata.index_from, 0, i)
      );
      const input = inputBuffer.toTensor();
      const predictOut = model.predict(input);
      const sentiment = predictOut.dataSync()[0];
      predictOut.dispose();
      return sentiment;
    };

    disableInput = false;
  });
</script>

<main>
  <h1>Svelte + LayerCake + D3</h1>
  <h2>Sentiment Analysis with TensorFlow.js</h2>

  <div>
    <textarea
      bind:value={textValue}
      cols={60}
      rows={7}
      disabled={disableInput}
    />
    <p>Sentiment : {sentiment}</p>
  </div>

  <div class="chart-container">
    <LayerCake
      padding={{ top: 8, right: 10, bottom: 20, left: 25 }}
      x={xKey}
      y={yKey}
      yNice={4}
      yDomain={[0, 1]}
      {data}
    >
      <Svg>
        <AxisY ticks={4} />
        <Line stroke={cubehelixInterpolator($tweenedSentiment)} />
        <Area fill={cubehelixInterpolator($tweenedSentiment)} opacity={0.2} />
      </Svg>
    </LayerCake>
  </div>
</main>

<style>
  .chart-container {
    width: 100%;
    height: 250px;
  }
</style>
