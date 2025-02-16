<script lang="ts" xmlns="http://www.w3.org/1999/html">
	let text: string;
	let test = [
		{ word: 'happy', value: 0.8 },
		{ word: 'sad', value: -0.6 },
		{ word: 'neutral', value: 0.1 }
	];
	let predictions: string;

	async function getText() {
		const response = await fetch('http://127.0.0.1:8000/analyze/', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ text })
		});
		predictions = await response.json();
	}
</script>

<div class="flex justify-center bg-[#1e1f22] text-white">
	<div class="m-1 flex h-20 w-2/4 items-center justify-center p-3 text-3xl font-bold">
		Sentiment Analysis
	</div>
</div>

<div class="flex w-full flex-col items-center gap-2 p-2">
	<div class="w-5/6 gap-3">
		<p class="mb-1.5">Enter text here:</p>
		<textarea
			bind:value={text}
			class="h-48 w-full rounded-lg bg-blue-300 p-2 shadow-md shadow-blue-600/50"
		></textarea>
		<div class="mt-2 flex w-full justify-end p-1">
			<button class="btn" on:click={getText}>Analyze</button>
		</div>
	</div>
</div>

<div class="flex w-full flex-col items-center gap-2 p-2">
	<div class="w-5/6 gap-3">
		<div class="mb-1.5">Model prediction: <strong>{predictions}</strong></div>
		<div
			class="flex min-h-24 w-full flex-wrap gap-3 rounded-lg bg-blue-300 p-2 shadow-md shadow-blue-600/50"
		>
			{#each test as p}
				<div class="flex w-full justify-around gap-2 p-1">
					<div
						class="max-w-56"
						class:positive={p.value >= 0.8}
						class:neutral={p.value > 0.4 && p.value < 0.8}
						class:negative={p.value < 0.4}
					>
						{p.word}
					</div>
					<div
						class="max-w-56"
						class:positive={p.value >= 0.8}
						class:neutral={p.value > 0.4 && p.value < 0.8}
						class:negative={p.value < 0.4}
					>
						{p.value}
					</div>
				</div>
			{/each}
		</div>
	</div>
</div>
