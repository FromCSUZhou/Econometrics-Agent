<script>
	import { goto } from '$app/navigation';
	import { userSignIn, userSignUp } from '$lib/apis/auths';
	import { WEBUI_API_BASE_URL, WEBUI_BASE_URL } from '$lib/constants';
	import { WEBUI_NAME, config, user } from '$lib/stores';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	let loaded = false;
	let mode = 'signin';

	let name = '';
	let email = '';
	let password = '';

	const setSessionUser = async (sessionUser) => {
		if (sessionUser) {
			console.log(sessionUser);
			toast.success(`You're now logged in.`);
			localStorage.token = sessionUser.token;
			await user.set(sessionUser);
			goto('/');
		}
	};

	const signInHandler = async () => {
		const sessionUser = await userSignIn(email, password).catch((error) => {
			toast.error(error);
			return null;
		});

		await setSessionUser(sessionUser);
	};

	const signUpHandler = async () => {
		const sessionUser = await userSignUp(name, email, password).catch((error) => {
			toast.error(error);
			return null;
		});

		await setSessionUser(sessionUser);
	};

	const submitHandler = async () => {
		if (mode === 'signin') {
			await signInHandler();
		} else {
			await signUpHandler();
		}
	};

	onMount(async () => {
		if ($user !== undefined) {
			await goto('/');
		}
		loaded = true;
	});
</script>

<svelte:head>
	<title>
		{`${$WEBUI_NAME}`}
	</title>
</svelte:head>

{#if loaded}
	<div class="fixed m-10 z-50">
		<div class="flex space-x-2">
			<div class=" self-center">
				<img src="{WEBUI_BASE_URL}/static/favicon.png" class=" w-8 rounded-full" alt="logo" />
			</div>
		</div>
	</div>

	<div class=" bg-white dark:bg-gray-900 min-h-screen w-full flex justify-center font-mona">

		<div class="w-full sm:max-w-lg px-4 min-h-screen flex flex-col">
			<div class=" my-auto pb-10 w-full">
				<form
					class=" flex flex-col justify-center bg-white py-6 sm:py-16 px-6 sm:px-16 rounded-2xl"
					on:submit|preventDefault={() => {
						submitHandler();
					}}
				>
					<div class=" text-xl sm:text-2xl font-bold">
						{mode === 'signin' ? 'Sign in' : 'Sign up'} to {$WEBUI_NAME}
					</div>

					{#if mode === 'signup'}
						<div class=" mt-1 text-xs font-medium text-gray-500">
							ⓘ {$WEBUI_NAME}: your data is encrypted for security.
						</div>
					{/if}

					<div class="flex flex-col mt-4">
						{#if mode === 'signup'}
							<div>
								<div class=" text-sm font-semibold text-left mb-1">Name</div>
								<input
									bind:value={name}
									type="text"
									class=" border px-4 py-2.5 rounded-2xl w-full text-sm"
									autocomplete="name"
									placeholder="Enter Your Full Name"
									required
								/>
							</div>

							<hr class=" my-3" />
						{/if}

						<div class="mb-2">
							<div class=" text-sm font-semibold text-left mb-1">Email</div>
							<input
								bind:value={email}
								type="email"
								class=" border px-4 py-2.5 rounded-2xl w-full text-sm"
								autocomplete="email"
								placeholder="Enter Your Email"
								required
							/>
						</div>

						<div>
							<div class=" text-sm font-semibold text-left mb-1">Password</div>
							<input
								bind:value={password}
								type="password"
								class=" border px-4 py-2.5 rounded-2xl w-full text-sm"
								placeholder="Enter Your Password"
								autocomplete="current-password"
								required
							/>
						</div>
					</div>

					<div class="mt-5">
						<button
							class=" bg-gray-900 hover:bg-gray-800 w-full rounded-full text-white font-semibold text-sm py-3 transition"
							type="submit"
						>
							{mode === 'signin' ? 'Sign In' : 'Create Account'}
						</button>

						<div class=" mt-4 text-sm text-center">
							<!-- {mode === 'signin' ? `Don't have an account?` : `Already have an account?`} -->

							<button
								class=" font-medium underline"
								type="button"
								on:click={() => {
									if (mode === 'signin') {
										mode = 'signup';
									} else {
										mode = 'signin';
									}
								}}
							>
								{mode === 'signin' ? `Sign up` : `Sign In`}
							</button>
						</div>
					</div>
				</form>
			</div>
		</div>
	</div>
{/if}

<style>
	.font-mona {
		font-family: 'Mona Sans';
	}
</style>
