<script lang="ts">
	let text;

	async function getText() {
		const response = await fetch('http://127.0.0.1:8000/analyze/', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ text })
		});
		const data = await response.json();
	}
</script>

<div class="flex justify-center bg-[#1e1f22] text-white">
	<div class="m-1 flex h-20 w-2/4 items-center justify-center p-3 text-3xl font-bold">
		Sentiment Analysis
	</div>
</div>

<div class="flex w-full flex-col items-center gap-2 p-3">
	<div class="w-5/6 gap-3 p-3">
		<p class="mb-1.5">Enter text here:</p>
		<textarea
			bind:value={text}
			class="h-48 w-full rounded-lg bg-blue-300 p-2 shadow-md shadow-blue-600/50"
		></textarea>
		<div class="mt-2 flex w-full justify-end">
			<button class="btn" on:click={getText}>Analyze</button>
		</div>
	</div>
</div>
