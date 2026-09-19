import ImageUpload from "./components/ImageUpload";

export default function Home() {
  return (
    <div className="flex flex-1 flex-col items-center bg-zinc-50 px-4 py-8 font-sans sm:px-6 sm:py-10">
      <main className="flex w-full max-w-[46rem] flex-col gap-5">
        <header className="text-center sm:text-left">
          <p className="text-sm font-medium tracking-wide text-emerald-700">
            FoliAI
          </p>
          <h1 className="mt-1.5 text-3xl font-semibold tracking-tight text-zinc-900">
            Plant disease classifier
          </h1>
          <p className="mt-2 max-w-2xl text-base leading-relaxed text-zinc-600">
            Identify common leaf diseases from a photo using FoliAI&apos;s
            computer vision model.
          </p>

          <div className="mt-3 flex flex-wrap items-center justify-center gap-2 sm:justify-start">
            {[
              "15 classes",
              "3 crops",
              "Bell pepper",
              "Potato",
              "Tomato",
              "Hybrid CV model",
            ].map((label) => (
              <span
                key={label}
                className="rounded-full bg-white px-2.5 py-1 text-xs font-medium text-zinc-600 ring-1 ring-zinc-200"
              >
                {label === "Hybrid CV model" ? (
                  <span className="text-emerald-700">{label}</span>
                ) : (
                  label
                )}
              </span>
            ))}
          </div>
        </header>

        <ImageUpload />
      </main>
    </div>
  );
}
