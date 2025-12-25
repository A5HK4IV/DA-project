import React, { useState } from "react";
import { useForm } from "react-hook-form";
import { postPatient } from "../../api/PostPatient";
import toast, { Toaster } from "react-hot-toast";
import { SpinnerCircularFixed } from "spinners-react";

const NewPatient = () => {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm();

  const submit = async (data) => {
    await postPatient(data).then((response) => {
      if (response.result == 0) {
        toast.success(
          "All clear - the patient's assessment returned a healty result."
        );
      } else {
        toast.error("Alert - the model predicts a potential health issue.");
      }
    });
  };

  return (
    <div className="flex flex-col py-auto items-center mx-auto my-auto max-w-7xl w-2xl justify-center sm:h-screen">
      <form
        className="w-xl rounded-md shadow-md/20 px-5 py-5"
        onSubmit={handleSubmit(submit)}
      >
        <div className="mb-6">
          <label
            htmlFor="model"
            className="block mb-2.5 text-sm font-medium text-heading"
          >
            Select Model
          </label>

          <select
            id="model"
            className="rounded-sm bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 shadow-xs placeholder:text-body"
            {...register("model", { required: true })}
            required
          >
            <option value="KNN">KNN</option>
            <option value="SVM">SVM</option>
            <option value="DecisionTree">Decision Tree</option>
            <option value="RandomForest">Random Forest</option>
            <option value="XGBoost">XGBoost</option>
            <option value="NaiveBayes">Naive Bayes</option>
            <option value="MLP">MLP</option>
          </select>
        </div>
        <div className="grid gap-6 mb-6 md:grid-cols-2">
          <div>
            <label
              htmlFor="age"
              className="block mb-2.5 text-sm font-medium text-heading"
            >
              Age
            </label>
            <input
              type="number"
              id="age"
              className="rounded-sm bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 shadow-xs placeholder:text-body"
              placeholder="25"
              {...register("age", {
                required: true,
                min: 30,
                max: 75,
                setValueAs: (v) => Number(v),
              })}
              aria-invalid={errors.age ? "true" : "false"}
              required
            />
            {errors.age && (
              <p className="text-red-600">Age must be between 30 and 75</p>
            )}
          </div>
          <div>
            <label
              htmlFor="sex"
              className="block mb-2.5 text-sm font-medium text-heading"
            >
              Gender
            </label>
            <select
              id="sex"
              className="rounded-sm bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 shadow-xs placeholder:text-body"
              {...register("sex", {
                required: true,
                setValueAs: (v) => Number(v),
              })}
              required
            >
              <option value={0}>Male</option>
              <option value={1}>Female</option>
            </select>
          </div>
          <div>
            <label
              htmlFor="cp"
              className="block mb-2.5 text-sm font-medium text-heading"
            >
              Chest Pain
            </label>
            <select
              id="cp"
              className="rounded-sm bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 shadow-xs placeholder:text-body"
              {...register("cp", {
                required: true,
                setValueAs: (v) => Number(v),
              })}
              required
            >
              <option value={0}>No Pain</option>
              <option value={1}>Mild Discomfort</option>
              <option value={2}>Moderate Pain</option>
              <option value={3}>Severe Pain</option>
            </select>
          </div>
          <div>
            <label
              htmlFor="trestbps"
              className="block mb-2.5 text-sm font-medium text-heading"
            >
              Patients Resting Blood Pressure
            </label>
            <input
              type="number"
              id="trestbps"
              className="rounded-sm bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 shadow-xs placeholder:text-body"
              placeholder="130"
              {...register("trestbps", {
                required: true,
                min: 90,
                max: 200,
                setValueAs: (v) => Number(v),
              })}
              aria-invalid={errors.trestbps ? "true" : "false"}
              required
            />
            {errors.trestbps && (
              <p className="text-red-600">
                Blood Pressure must be between 90 and 200
              </p>
            )}
          </div>
          <div>
            <label
              htmlFor="chol"
              className="block mb-2.5 text-sm font-medium text-heading"
            >
              Cholesterol Levels
            </label>
            <input
              type="number"
              id="chol"
              className="rounded-sm bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 shadow-xs placeholder:text-body"
              placeholder="204"
              {...register("chol", {
                required: true,
                min: 130,
                max: 550,
                setValueAs: (v) => Number(v),
              })}
              aria-invalid={errors.chol ? "true" : "false"}
              required
            />
            {errors.chol && (
              <p className="text-red-600">
                Cholesterol levels must be between 130 and 550
              </p>
            )}
          </div>
          <div>
            <label
              htmlFor="fbs"
              className="block mb-2.5 text-sm font-medium text-heading"
            >
              Fasting Blood Sugar
            </label>
            <select
              id="fbs"
              className="rounded-sm bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 shadow-xs placeholder:text-body"
              {...register("fbs", {
                required: true,
                setValueAs: (v) => Number(v),
              })}
              required
            >
              <option value={0}>False</option>
              <option value={1}>True</option>
            </select>
          </div>
          <div>
            <label
              htmlFor="restecg"
              className="block mb-2.5 text-sm font-medium text-heading"
            >
              Resting Electrocardiographic
            </label>
            <select
              id="restecg"
              className="rounded-sm bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 shadow-xs placeholder:text-body"
              {...register("restecg", {
                required: true,
                setValueAs: (v) => Number(v),
              })}
              required
            >
              <option value={0}>Normal</option>
              <option value={1}>ST-T Wave Abnormality</option>
              <option value={2}>Left Ventricular Hypertrophy</option>
            </select>
          </div>
          <div>
            <label
              htmlFor="thalach"
              className="block mb-2.5 text-sm font-medium text-heading"
            >
              Maximum Heart Rate
            </label>
            <input
              type="number"
              id="thalach"
              className="rounded-sm bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 shadow-xs placeholder:text-body"
              placeholder="187"
              {...register("thalach", {
                required: true,
                min: 70,
                max: 200,
                setValueAs: (v) => Number(v),
              })}
              aria-invalid={errors.thalach ? "true" : "false"}
              required
            />
            {errors.thalach && (
              <p className="text-red-600">
                Maximum Heart Rate is not acceptable
              </p>
            )}
          </div>
          <div>
            <label
              htmlFor="exang"
              className="block mb-2.5 text-sm font-medium text-heading"
            >
              Exercise-Induced Angina
            </label>
            <select
              id="exang"
              className="rounded-sm bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 shadow-xs placeholder:text-body"
              {...register("exang", {
                required: true,
                setValueAs: (v) => Number(v),
              })}
              required
            >
              <option value={0}>False</option>
              <option value={1}>True</option>
            </select>
          </div>
          <div>
            <label
              htmlFor="oldpeak"
              className="block mb-2.5 text-sm font-medium text-heading"
            >
              Amount of ST Depression on ESG
            </label>
            <input
              type="number"
              step="0.1"
              id="oldpeak"
              className="rounded-sm bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 shadow-xs placeholder:text-body"
              placeholder="1.5"
              {...register("oldpeak", {
                required: true,
                min: 0,
                max: 6.5,
                setValueAs: (v) => Number(v),
              })}
              aria-invalid={errors.oldpeak ? "true" : "false"}
              required
            />
            {errors.oldpeak && (
              <p className="text-red-500">Must be between 0 and 6.5</p>
            )}
          </div>
          <div>
            <label
              htmlFor="slope"
              className="block mb-2.5 text-sm font-medium text-heading"
            >
              ST/HR Slope
            </label>
            <select
              id="slope"
              className="rounded-sm bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 shadow-xs placeholder:text-body"
              {...register("slope", {
                required: true,
                setValueAs: (v) => Number(v),
              })}
              required
            >
              <option value={0}>Upsloping</option>
              <option value={1}>Flat</option>
              <option value={2}>Downsloping</option>
            </select>
          </div>
          <div>
            <label
              htmlFor="ca"
              className="block mb-2.5 text-sm font-medium text-heading"
            >
              Colored Vessels by Fluoroscopy
            </label>
            <select
              id="ca"
              className="rounded-sm bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 shadow-xs placeholder:text-body"
              {...register("ca", {
                required: true,
                setValueAs: (v) => Number(v),
              })}
              required
            >
              <option value={0}>0</option>
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={3}>3</option>
              <option value={4}>4</option>
            </select>
          </div>
          <div>
            <label
              htmlFor="thal"
              className="block mb-2.5 text-sm font-medium text-heading"
            >
              Thalassemia Status
            </label>
            <select
              id="thal"
              className="rounded-sm bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 shadow-xs placeholder:text-body"
              {...register("thal", {
                required: true,
                setValueAs: (v) => Number(v),
              })}
              required
            >
              <option value={0}>Normal</option>
              <option value={1}>Fixed Defect</option>
              <option value={2}>Reversible Defect</option>
            </select>
          </div>
          <div className="content-end">
            <button
              type="submit"
              className="rounded-sm h-10 bg-neutral-secondary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 bg-sky-500 text-white shadow-xs placeholder:text-body hover:bg-sky-700"
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <SpinnerCircularFixed
                  className="mx-auto justify-center"
                  color="#ffffff"
                  size={20}
                />
              ) : (
                "Check Patient"
              )}
            </button>
            <Toaster />
          </div>
        </div>
      </form>
    </div>
  );
};

export default NewPatient;
